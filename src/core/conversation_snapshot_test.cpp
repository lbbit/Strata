#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "kv_pool.hpp"
#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <cstdlib>
#include <limits>
#include <vector>

using namespace strata::core;
using namespace strata::kernels;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
void cuda_check_at(cudaError_t e, int line) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA line %d: %s\n", line, cudaGetErrorString(e)); std::exit(1); }
}
#define cuda_check(e) cuda_check_at((e), __LINE__)
struct Fixture {
    ModelGeometry g;
    QsaState state;
    std::vector<void*> device, host;
    std::array<void*,5> sources{};
    std::array<size_t,5> sizes{};

    template<class T> void alloc(T*& p, size_t n, bool pinned = false) {
        if (!n) return;
        void* raw = nullptr;
        if (pinned) {
            cuda_check(cudaHostAlloc(&raw, n, cudaHostAllocMapped));
            host.push_back(raw);
            void* mapped = nullptr;
            cuda_check(cudaHostGetDevicePointer(&mapped, raw, 0));
            p = static_cast<T*>(mapped);
        } else {
            cuda_check(cudaMalloc(&raw, n)); device.push_back(raw); p = static_cast<T*>(raw);
        }
    }
    Fixture(int fmt, int mode) {
        auto& st = state;
        st.kv_mode = mode; st.kv_int8 = fmt==kKvInt8; st.kv_q4 = fmt==kKvQ4; st.kv_hybrid = fmt==3;
        st.n_pages = 24; st.n_slots = mode ? 4 : st.n_pages;
        st.max_cells = st.n_pages * 4; st.idx_pooled_rows = mode==2 ? 2 : st.max_cells/4+2;
        const size_t per = fmt==kKvQ4 ? kv_q4_bytes_per_head((int)g.head_dim) : g.head_dim*((st.kv_int8 || st.kv_hybrid) ? 1:2);
        const size_t rows = st.max_cells*g.n_head_kv, slot_rows = st.n_slots*4*g.n_head_kv;
        sizes = {rows*per, rows*per, fmt==kKvInt8 ? rows*(g.head_dim/64)*2:0,
                 fmt==kKvInt8 ? rows*(g.head_dim/64)*2:0, (size_t)st.idx_pooled_rows*g.idx_key_dim*4};
        if (fmt==3) {
            sizes = {rows*per, rows*kv_q4_bytes_per_head((int)g.head_dim), rows*(g.head_dim/64)*2, 0,
                     (size_t)st.idx_pooled_rows*g.idx_key_dim*4};
            alloc(st.k_q,sizes[0]); alloc(st.v_q4,sizes[1]); alloc(st.k_scale,sizes[2]);
            sources[0]=st.k_q; sources[1]=st.v_q4; sources[2]=st.k_scale;
        } else if (fmt==kKvQ4) {
            alloc(st.k_q4,slot_rows*per); alloc(st.v_q4,slot_rows*per);
            if (mode) {alloc(st.host.k_q4,sizes[0],true); alloc(st.host.v_q4,sizes[1],true);}
            sources[0]=mode?st.host.k_q4:st.k_q4; sources[1]=mode?st.host.v_q4:st.v_q4;
        } else if (fmt==kKvInt8) {
            alloc(st.k_q,slot_rows*per); alloc(st.v_q,slot_rows*per);
            alloc(st.k_scale,slot_rows*(g.head_dim/64)*2); alloc(st.v_scale,slot_rows*(g.head_dim/64)*2);
            if (mode) {
                alloc(st.host.k_q,sizes[0],true); alloc(st.host.v_q,sizes[1],true);
                alloc(st.host.k_scale,sizes[2],true); alloc(st.host.v_scale,sizes[3],true);
            }
            sources[0]=mode?st.host.k_q:st.k_q; sources[1]=mode?st.host.v_q:st.v_q;
            sources[2]=mode?st.host.k_scale:st.k_scale; sources[3]=mode?st.host.v_scale:st.v_scale;
        } else {
            alloc(st.k_pool,slot_rows*per); alloc(st.v_pool,slot_rows*per);
            if (mode) {alloc(st.host.k_pool,sizes[0],true); alloc(st.host.v_pool,sizes[1],true);}
            sources[0]=mode?st.host.k_pool:st.k_pool; sources[1]=mode?st.host.v_pool:st.v_pool;
        }
        alloc(st.idx_pooled,sizes[4]); sources[4]=st.idx_pooled;
        alloc(st.page_table,st.n_pages*4);
        if (mode==1) {
            auto& m=st.map;
            m.page_table=st.page_table; m.n_blocks=st.n_pages; m.n_slots=st.n_slots;
            alloc(m.slot_block,st.n_slots*4); alloc(m.slot_stamp,st.n_slots*4); alloc(m.slot_ref,st.n_slots*4);
            alloc(m.miss_block,st.n_slots*4); alloc(m.miss_slot,st.n_slots*4); alloc(m.ctl,kKvCtlInts*4);
        } else if (mode==2) {
            kv_ring_table(st.page_table,st.n_pages,st.n_slots,nullptr);
        }
    }
    void fill(uint8_t salt) {
        for (size_t i=0;i<sources.size();++i) {
            std::vector<uint8_t> data(sizes[i]);
            for (size_t j=0;j<data.size();++j) data[j]=(uint8_t)(salt+i*31+j*7+j/257);
            if (!data.empty()) cuda_check(cudaMemcpy(sources[i],data.data(),data.size(),cudaMemcpyDefault));
        }
    }
    void fill_after(uint8_t salt, int64_t first_dirty) {
        for (size_t i=0;i<sources.size();++i) {
            const size_t offset = i == 4 ? size_t(first_dirty/4)*g.idx_key_dim*4
                                         : (sizes[i]/size_t(state.max_cells))*size_t((first_dirty/4)*4);
            if (sizes[i] > offset)
                cuda_check(cudaMemset(static_cast<uint8_t*>(sources[i])+offset, salt, sizes[i]-offset));
        }
    }
    ~Fixture() { for (void* p:device) cudaFree(p); for (void* p:host) cudaFreeHost(p); }
};
bool equal(const ConversationKv& a,const ConversationKv& b) {
    return a.k==b.k && a.v==b.v && a.k_scale==b.k_scale && a.v_scale==b.v_scale && a.pooled==b.pooled;
}

struct PooledFixture {
    ModelGeometry g;
    std::unique_ptr<GlobalKvPool> pool;
    QsaState a, b, c;
    std::vector<int32_t*> tables;
    std::vector<float*> indexers;
    std::string err;

    PooledFixture(int fmt, int pages) {
        const int64_t price=(int64_t)part_price(fmt);
        pool=GlobalKvPool::create(0,g,1,pages*price,err,{},format_of(fmt));
        check(pool!=nullptr,"create real GPU pool for snapshot fixture");
        a=state(fmt,pages); b=state(fmt,pages); c=state(fmt,pages);
        for (QsaState* st : {&a,&b,&c}) {
            pool->bind_main_state(*st);
            tables.push_back(st->page_table); indexers.push_back(st->idx_pooled);
        }
    }
    static KvPoolFormat format_of(int fmt) {
        return fmt==kKvInt8?KvPoolFormat::Int8:fmt==kKvQ4?KvPoolFormat::Q4:fmt==3?KvPoolFormat::Hybrid:KvPoolFormat::Fp16;
    }
    static size_t part_price(int fmt) {
        ModelGeometry g;
        uint64_t n=0; std::string e;
        check(kv_pool_page_price(qsa_shapes(g),format_of(fmt),n,e),"snapshot fixture page price");
        return n;
    }
    QsaState state(int fmt,int pages) {
        QsaState st;
        st.kv_int8=fmt==kKvInt8; st.kv_q4=fmt==kKvQ4; st.kv_hybrid=fmt==3;
        st.n_pages=st.n_slots=pages; st.max_cells=pages*4; st.idx_pooled_rows=pages+2;
        cuda_check(cudaMalloc((void**)&st.page_table,st.n_pages*sizeof(int32_t)));
        cuda_check(cudaMalloc((void**)&st.idx_pooled,st.idx_pooled_rows*g.idx_key_dim*sizeof(float)));
        return st;
    }
    QsaState& nth(int n) { return n==0?a:n==1?b:c; }
    void mapping(QsaState& st,int pages,uint32_t seed) {
        check(st.kv_pool_page_mirror!=nullptr,"live host mirror for fixed mapping");
        auto& table=*st.kv_pool_page_mirror;
        std::vector<int32_t> permutation((size_t)pages);
        for (int i=0;i<pages;++i) permutation[(size_t)i]=i;
        for (int i=pages-1;i>0;--i) {
            seed=seed*1664525u+1013904223u;
            std::swap(permutation[(size_t)i],permutation[(size_t)(seed%(uint32_t)(i+1))]);
        }
        std::vector<int32_t> old(table.begin(),table.begin()+pages);
        for (int i=0;i<pages;++i) table[(size_t)i]=old[(size_t)permutation[(size_t)i]];
        cuda_check(cudaMemcpy(st.page_table,table.data(),table.size()*sizeof(int32_t),cudaMemcpyHostToDevice));
    }
    size_t part_page_bytes(int fmt,int part) const {
        if (part<2) {
            const size_t per=fmt==kKvQ4||fmt==3&&part==1 ? kv_q4_bytes_per_head((int)g.head_dim)
                                                          : g.head_dim*((fmt==kKvInt8||fmt==3)?1:2);
            return 4*(size_t)g.n_head_kv*per;
        }
        if (part<4) return (part==2 && (fmt==kKvInt8||fmt==3)) || (part==3 && fmt==kKvInt8)
                            ? 4*(size_t)g.n_head_kv*(g.head_dim/64)*2 : 0;
        return 0;
    }
    void* part_base(QsaState& st,int fmt,int part) {
        if (fmt==3) return part==0?(void*)st.k_q:part==1?(void*)st.v_q4:part==2?(void*)st.k_scale:nullptr;
        if (fmt==kKvQ4) return part==0?(void*)st.k_q4:part==1?(void*)st.v_q4:nullptr;
        if (fmt==kKvInt8) return part==0?(void*)st.k_q:part==1?(void*)st.v_q:part==2?(void*)st.k_scale:part==3?(void*)st.v_scale:nullptr;
        return part==0?(void*)st.k_pool:part==1?(void*)st.v_pool:nullptr;
    }
    void fill_parts(QsaState& st,int fmt,int pages,uint8_t salt) {
        std::vector<int32_t> table;
        check(pool->snapshot_page_table(st,table,err),"read fixed mapping for pattern upload");
        for (int part=0;part<4;++part) {
            const size_t page_bytes=part_page_bytes(fmt,part);
            if (!page_bytes) continue;
            std::vector<uint8_t> logical((size_t)pages*page_bytes);
            for (size_t i=0;i<logical.size();++i) logical[i]=(uint8_t)(salt+part*31+i*7+i/251);
            for (int p=0;p<pages;++p)
                cuda_check(cudaMemcpy((uint8_t*)part_base(st,fmt,part)+(size_t)table[(size_t)p]*page_bytes,
                                      logical.data()+(size_t)p*page_bytes,page_bytes,cudaMemcpyHostToDevice));
        }
        const size_t index_bytes=(size_t)(pages+1)*g.idx_key_dim*sizeof(float);
        std::vector<uint8_t> index(index_bytes);
        for (size_t i=0;i<index.size();++i) index[i]=(uint8_t)(salt+157+i*7+i/251);
        cuda_check(cudaMemcpy(st.idx_pooled,index.data(),index.size(),cudaMemcpyHostToDevice));
        cuda_check(cudaDeviceSynchronize());
    }
    size_t k_page_bytes(int fmt) const { return part_page_bytes(fmt,0); }
    void* k_base(QsaState& st,int fmt) { return part_base(st,fmt,0); }
    void fill_k(QsaState& st,int fmt,int pages,uint8_t salt) { fill_parts(st,fmt,pages,salt); }
    ~PooledFixture() {
        for (void* p:tables) (void)cudaFree(p);
        for (void* p:indexers) (void)cudaFree(p);
    }
};

void pooled_snapshots(int fmt) {
    constexpr int pages=18, capacity=128;
    PooledFixture f(fmt,capacity);
    for (int n=0;n<3;++n) check(f.pool->ensure(f.nth(n),0,pages*4,nullptr,f.err),"allocate small pooled state");
    for (int n=0;n<3;++n) f.mapping(f.nth(n),pages,17+(uint32_t)n);
    std::vector<int32_t> map_a,map_b;
    check(f.pool->snapshot_page_table(f.a,map_a,f.err) && f.pool->snapshot_page_table(f.b,map_b,f.err) && map_a!=map_b,
          "source and target have different fixed nonidentity mappings");
    bool nonidentity=false;
    for (int i=0;i<pages;++i) nonidentity |= map_a[(size_t)i]!=i;
    check(nonidentity,"test fixture genuinely uses a nonidentity map");
    f.fill_k(f.a,fmt,pages,(uint8_t)(19*fmt));

    ConversationKv image;
    check(conversation_kv_save(image,f.a,f.g,pages*4,true,f.err),"gather fixed nonidentity pages");
    check(image.k.size()==(size_t)pages*f.k_page_bytes(fmt),"identity-layout snapshot byte size unchanged");
    std::vector<uint8_t> expected(image.k.size()),got(expected.size());
    for (size_t i=0;i<expected.size();++i) expected[i]=(uint8_t)(19*fmt+i*7+i/251);
    check(image.k.read(got.data(),0,got.size()) && got==expected,"pooled gather equals logical bytes");

    // Restore the same logical bytes into a differently mapped state. This is
    // the layout-invariance check: merely permuting A's table would change its
    // logical contents and is not a valid comparison.
    check(conversation_kv_restore(image,f.b,f.g,pages*4,true,f.err),"restore same logical bytes into another mapping");
    ConversationKv same;
    check(conversation_kv_save(same,f.b,f.g,pages*4,true,f.err),"capture second physical layout");
    check(equal(image,same),"snapshot independent of physical layout");
    uint64_t fp_a=0,fp_b=0;
    check(conversation_kv_verify(image,f.a,f.g,pages*4,true,fp_a,f.err) &&
          conversation_kv_verify(image,f.b,f.g,pages*4,true,fp_b,f.err) && fp_a==fp_b,
          "fingerprint independent of physical layout");

    SessionKvSource source;
    check(conversation_kv_source(source,f.a,f.g,pages*4,true,f.err),"SAVE source with fixed mapping");
    for (size_t part=0;part<5;++part) {
        const std::array<const ConversationBuffer*,5> parts={&image.k,&image.v,&image.k_scale,&image.v_scale,&image.pooled};
        std::vector<uint8_t> out(source.sizes[part]),want(source.sizes[part]);
        const size_t page_bytes=part<4?(size_t)source.sizes[part]/(size_t)pages:source.sizes[part];
        const size_t first=out.empty()?0:std::min(out.size()-1,page_bytes-7);
        check(source.read(part,0,out.data(),first) && source.read(part,first,out.data()+first,out.size()-first),
              "small SAVE reads cross a logical page boundary");
        check(parts[part]->read(want.data(),0,want.size()) && out==want,"SAVE source equals logical snapshot");
    }
    cuda_check(cudaDeviceSynchronize()); ConversationKv incremental=image;
    size_t reused=0;
    check(cudaGetLastError()==cudaSuccess,"clear prior asynchronous errors before incremental capture");
    check(conversation_kv_save(incremental,f.a,f.g,pages*4,true,f.err,7,&reused) && equal(incremental,image) && reused>0,
          "unchanged_tokens inside a page preserves full logical capture");
    for (bool queued:{false,true}) {
        check(conversation_kv_copy(f.a,f.b,f.g,pages*4,true,f.err,queued),"pooled direct copy queued false/true");
        cuda_check(cudaDeviceSynchronize());
        ConversationKv copied;
        check(conversation_kv_save(copied,f.b,f.g,pages*4,true,f.err),"capture direct-copy target");
        check(equal(copied,image),"direct copy preserves logical bytes in target mapping");
    }
    check(conversation_kv_restore(image,f.c,f.g,pages*4,true,f.err),"scatter restore into fresh pooled state");
    ConversationKv restored;
    check(conversation_kv_save(restored,f.c,f.g,pages*4,true,f.err),"capture restored QSA state");
    check(equal(restored,image),"restore roundtrip");
    check(conversation_kv_restore_prefix(image,f.c,f.g,pages*4,67,true,f.err),"restore non-page-aligned prefix");
    ConversationKv actual_prefix;
    check(conversation_kv_save(actual_prefix,f.c,f.g,67,true,f.err),"capture nonaligned restored prefix");
    const std::array<const ConversationBuffer*,5> got_prefix={&actual_prefix.k,&actual_prefix.v,&actual_prefix.k_scale,&actual_prefix.v_scale,&actual_prefix.pooled};
    const std::array<const ConversationBuffer*,5> captured={&image.k,&image.v,&image.k_scale,&image.v_scale,&image.pooled};
    for (size_t part=0;part<5;++part) {
        const size_t page_bytes=part<4?actual_prefix.k.size()/(size_t)((67+3)/4):0;
        const size_t bytes=part==4?(size_t)actual_prefix.pooled_rows*f.g.idx_key_dim*sizeof(float)
                                  :page_bytes?got_prefix[part]->size():0;
        std::vector<uint8_t> got(bytes),want(bytes);
        check(got_prefix[part]->read(got.data(),0,bytes) && captured[part]->read(want.data(),0,bytes) && got==want,
              "nonaligned restore_prefix copies the exact logical prefix");
    }

    std::vector<int32_t> before_a,before_b;
    check(f.pool->snapshot_page_table(f.a,before_a,f.err) && f.pool->snapshot_page_table(f.b,before_b,f.err),
          "capture maps before exhaustion");
    ConversationKv old_a,old_b;
    check(conversation_kv_save(old_a,f.a,f.g,pages*4,true,f.err) &&
          conversation_kv_save(old_b,f.b,f.g,pages*4,true,f.err),"capture old bytes before exhaustion");
    QsaState filler=f.state(fmt,capacity-3*pages);
    check(f.pool->ensure(filler,0,(capacity-3*pages)*4,nullptr,f.err),"fill every remaining pool page");
    const int64_t free_before=f.pool->free_pages();
    check(!conversation_kv_restore_many({&image,&image,&image},{&f.a,&f.b,&f.c},f.g,pages*4,pages*4,true,f.err),
          "third-layer prepare exhausts pool");
    std::vector<int32_t> after_a,after_b;
    check(f.pool->snapshot_page_table(f.a,after_a,f.err) && f.pool->snapshot_page_table(f.b,after_b,f.err) &&
          after_a==before_a && after_b==before_b && f.pool->free_pages()==free_before,
          "failed prepare preserves maps and free count");
    ConversationKv now_a,now_b;
    check(conversation_kv_save(now_a,f.a,f.g,pages*4,true,f.err) &&
          conversation_kv_save(now_b,f.b,f.g,pages*4,true,f.err) && equal(now_a,old_a) && equal(now_b,old_b),
          "failed prepare preserves all old layer bytes");
    check(f.pool->release(filler,nullptr,f.err),"release exhaustion filler");
    (void)cudaFree(filler.page_table); (void)cudaFree(filler.idx_pooled);

    std::vector<int32_t> retired=before_a;
    check(f.pool->release(f.a,nullptr,f.err) && !f.pool->owns(f.a),"released conversation has no stale lease");
    check(f.pool->ensure(f.a,0,4,nullptr,f.err),"recycle a physical page for another conversation");
    cuda_check(cudaDeviceSynchronize());
    const int32_t physical=f.pool->physical_page(f.a,0);
    check(std::find(retired.begin(),retired.end(),physical)!=retired.end(),"new conversation really reuses an old physical page");
    std::vector<uint8_t> padding(f.k_page_bytes(fmt),0xff);
    cuda_check(cudaMemcpy(padding.data(),(uint8_t*)f.k_base(f.a,fmt)+(size_t)physical*f.k_page_bytes(fmt),
                          padding.size(),cudaMemcpyDeviceToHost));
    check(std::all_of(padding.begin(),padding.end(),[](uint8_t v){return v==0;}),"reused physical page has no stale padding");
}

void pooled_segment_crossing() {
    // 4096 FP16 K pages are exactly 16 MiB. Use 4100 so one host interval
    // crosses both the 4-cell physical-page boundary and the 16 MiB segment.
    constexpr int pages=4100;
    PooledFixture f(kKvF16,3*pages+8);
    check(f.pool->ensure(f.a,0,pages*4,nullptr,f.err) && f.pool->ensure(f.b,0,pages*4,nullptr,f.err),
          "allocate two large pooled states");
    f.mapping(f.a,pages,101); f.mapping(f.b,pages,202);
    f.fill_k(f.a,kKvF16,pages,37);
    ConversationKv image;
    check(conversation_kv_save(image,f.a,f.g,pages*4,true,f.err),"gather across 16 MiB segment boundary");
    check(image.k.size()>(16u<<20),"large K payload really crosses ConversationBuffer segments");
    const bool large_restore=conversation_kv_restore(image,f.b,f.g,pages*4,true,f.err);
    check(large_restore,"scatter across 16 MiB segment boundary");
    ConversationKv restored;
    check(conversation_kv_save(restored,f.b,f.g,pages*4,true,f.err) && equal(image,restored),
          "large gather/scatter roundtrip is byte exact");
    SessionKvSource source;
    check(conversation_kv_source(source,f.a,f.g,pages*4,true,f.err),"large SAVE source");
    const size_t page_bytes=f.k_page_bytes(kKvF16);
    const size_t offset=(16u<<20)-page_bytes-7;
    const size_t bytes=2*page_bytes+14;
    std::vector<uint8_t> streamed(bytes),captured(bytes);
    check(source.read(0,offset,streamed.data(),streamed.size()) && image.k.read(captured.data(),offset,captured.size()) &&
          streamed==captured,"one SAVE read crosses both a page and the 16 MiB segment boundary");
    std::vector<uint8_t> all(source.sizes[0]),want(source.sizes[0]);
    check(source.read(0,0,all.data(),all.size()) && image.k.read(want.data(),0,want.size()) && all==want,
          "whole segmented SAVE stream equals captured logical bytes");
}

void full_session(int fmt, int mode, int experts) {
    Fixture main(fmt,mode), draft(fmt==3?kKvInt8:fmt,2);
    auto& g=main.g;
    g.n_layers=4; g.n_expert=experts;
    g.ssm_state_size=2; g.ssm_v_heads=2; g.ssm_conv_channels=8;
    SessionState ss;
    ss.max_cells=96; ss.qsa_states=&main.state;
    ss.layer_hi=g.n_layers; ss.gdn_alloc=g.n_gdn_layers(); ss.qsa_alloc=g.n_qsa_layers();   // the whole-model carve
    std::string err;
    ConversationStateSizes sizes;
    check(conversation_state_sizes(g,sizes,err),"whole-session geometry sizes");
    main.alloc(ss.gdn_state,sizes.gdn); main.alloc(ss.ple_hist,sizes.ple);
    main.alloc(main.state.idx_tail,sizes.tail); main.alloc(main.state.idx_dead,sizes.dead);
    main.alloc(main.state.idx_block_pos,sizes.block_pos);
    std::vector<int32_t> ids(65);
    for (size_t i=0;i<ids.size();++i) ids[i]=(int32_t)i+1;
    std::vector<ConversationImageKey> images;
    std::vector<ConversationCheckpoint> checkpoints;
    auto fill=[&](uint8_t salt) {
        main.fill(salt); draft.fill(salt);
        for (const auto& [p,n] : std::vector<std::pair<void*,size_t>>{
                 {ss.gdn_state,sizes.gdn},{ss.ple_hist,sizes.ple},{main.state.idx_tail,sizes.tail},
                 {main.state.idx_dead,sizes.dead},{main.state.idx_block_pos,sizes.block_pos}})
            cuda_check(cudaMemset(p,salt,n));
        // A real indexer maintains its spare pooled row as a copy of idx_dead.
        cuda_check(cudaMemcpy(main.state.idx_pooled+(ids.size()/4)*g.idx_key_dim,
                              main.state.idx_dead,sizes.dead,cudaMemcpyDeviceToDevice));
        cuda_check(cudaDeviceSynchronize());
    };
    fill(13);
    ConversationCheckpoint checkpoint;
    checkpoint.ids.assign(ids.begin(),ids.begin()+3);
    check(conversation_checkpoint_save(checkpoint,ss,g,err),"save complete running checkpoint");
    checkpoint.used = 17;
    checkpoints.push_back(std::move(checkpoint));
    const ConversationView view{ids,images,checkpoints,true};
    SavedConversation a,b,restored;
    check(conversation_snapshot_save(a,view,ss,g,draft.state,err),"capture complete A");
    check(a.checkpoints[0].used == 17,"upstream checkpoint LRU stamp survives capture");
    fill(177);
    check(conversation_snapshot_save(b,view,ss,g,draft.state,err),"capture complete B");
    {
        // disk save path: metadata + streamed K/V give the same file as the captured image
        SavedConversation meta;
        std::vector<SessionKvSource> sources;
        check(conversation_snapshot_sources(meta,sources,view,ss,g,draft.state,err),"disk-save sources");
        check(meta.kv.empty() && sources.size()==b.kv.size(),"one source per K/V layer, none captured");
        const SessionFileIdentity id{1,2};
        namespace fs=std::filesystem;
        const fs::path dir=fs::temp_directory_path()/("strata-snap-test-"+std::to_string(experts)+"-"+
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        fs::create_directories(dir);
        const std::string p1=(dir/"streamed.bin").string(), p2=(dir/"captured.bin").string();
        auto slurp=[](const std::string& p){ std::ifstream f(p,std::ios::binary);
            return std::vector<char>((std::istreambuf_iterator<char>(f)),std::istreambuf_iterator<char>()); };
        size_t n1=0,n2=0;
        check(session_file_write(p1,meta,sources,id,n1,err),"streamed session write");
        check(session_file_write(p2,b,id,n2,err),"captured session write");
        SavedConversation x,y;
        check(session_file_read(p1,id,x,n1,err) && session_file_read(p2,id,y,n2,err) && n1==n2,"both files read");
        check(x.live.gdn==y.live.gdn && x.live.dead==y.live.dead && x.live.ids==y.live.ids &&
              x.checkpoints.size()==y.checkpoints.size() && equal(x.kv[0],y.kv[0]) && equal(x.kv.back(),y.kv.back()),
              "streamed file equals captured file");
        check(slurp(p1)==slurp(p2),"streamed file is byte-identical to the captured file");
        std::error_code ec;
        fs::remove_all(dir,ec);
        auto empty_ids=std::vector<int32_t>{};
        const ConversationView none{empty_ids,images,checkpoints,true};
        check(!conversation_snapshot_sources(meta,sources,none,ss,g,draft.state,err),"empty session refused");
    }
    check(a.live.dead!=b.live.dead,"different opening-state spare keys in regression fixture");
    auto bad=a; bad.kv.back().k.pop_back();
    check(conversation_snapshot_restore(bad,ss,g,draft.state,err)==ConversationRestore::invalid,
          "reject invalid late draft before any main-layer write");
    check(conversation_snapshot_save(restored,view,ss,g,draft.state,err),"capture B after refused restore");
    check(restored.live.gdn==b.live.gdn && restored.live.dead==b.live.dead &&
          equal(restored.kv[0],b.kv[0]) && equal(restored.kv[1],b.kv[1]),"refusal preserves all tested state");
    check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore complete A");
    check(conversation_snapshot_save(restored,view,ss,g,draft.state,err),"capture restored A");
    check(restored.live.gdn==a.live.gdn && restored.live.ple==a.live.ple && restored.live.tails==a.live.tails &&
          restored.live.dead==a.live.dead && restored.live.block_pos==a.live.block_pos &&
          equal(restored.kv[0],a.kv[0]) && equal(restored.kv[1],a.kv[1]),"whole-session A/B/A exactness including spare key");
    check(ss.ple_prev[0]==64 && ss.ple_prev[1]==65,"PLE token window reconstructed");
    {
        // a borrowed prefix (ConversationCache::borrows): the K/V of A's first tokens and A's checkpoint there, A
        // untouched - the same bytes as restoring A whole and rewinding to that checkpoint
        const int64_t at = (int64_t) a.checkpoints[0].ids.size();
        check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore A whole");
        check(conversation_checkpoint_restore(a.checkpoints[0],ss,g,err),"rewind it to its checkpoint");
        ConversationKv want_main,want_draft,got_main,got_draft;
        ConversationCheckpoint want_state,got_state;
        want_state.ids=got_state.ids=a.checkpoints[0].ids;
        check(conversation_kv_save(want_main,main.state,g,at,true,err) &&
              conversation_kv_save(want_draft,draft.state,g,at,false,err) &&
              conversation_checkpoint_save(want_state,ss,g,err),"capture the rewound prefix");
        fill(201);   // another conversation in the session
        const SavedConversation untouched=a;
        check(conversation_snapshot_restore_prefix(a,(int64_t)a.live.ids.size()+1,ss,g,draft.state,err)==
                  ConversationRestore::invalid &&
              conversation_snapshot_restore_prefix(a,0,ss,g,draft.state,err)==ConversationRestore::invalid,
              "a prefix outside the image is refused");
        check(conversation_snapshot_restore_prefix(a,at,ss,g,draft.state,err)==ConversationRestore::restored,
              "restore A's prefix only");
        check(conversation_checkpoint_restore(a.checkpoints[0],ss,g,err),"mount the checkpoint over it");
        check(conversation_kv_save(got_main,main.state,g,at,true,err) &&
              conversation_kv_save(got_draft,draft.state,g,at,false,err) &&
              conversation_checkpoint_save(got_state,ss,g,err),"capture the borrowed prefix");
        check(equal(got_main,want_main) && equal(got_draft,want_draft),
              "borrowed K/V equals the whole restore's prefix");
        check(got_state.gdn==want_state.gdn && got_state.ple==want_state.ple && got_state.tails==want_state.tails &&
              got_state.dead==want_state.dead && got_state.block_pos==want_state.block_pos,"the same running state");
        check(untouched.live.ids==a.live.ids && equal(untouched.kv[0],a.kv[0]) && equal(untouched.kv[1],a.kv[1]),
              "the parked image is unchanged");
    }
    for (int64_t dirty : {65, 3, 0}) {
        check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore growth fixture base");
        ConversationKvReuse reuse{a.kv,65,dirty};
        const uint8_t* original = nullptr;
        reuse.kv[0].k.visit(0,1,[&](const uint8_t* p,size_t,size_t){original=p;return true;});
        main.fill_after(91,dirty); draft.fill_after(91,std::max<int64_t>(0,dirty-1));
        cuda_check(cudaMemset(ss.gdn_state,91,sizes.gdn));
        ids.resize(70);
        for (size_t i=65;i<ids.size();++i) ids[i]=int32_t(i+1);
        cuda_check(cudaMemcpy(main.state.idx_pooled+(ids.size()/4)*g.idx_key_dim,
                              main.state.idx_dead,sizes.dead,cudaMemcpyDeviceToDevice));
        SavedConversation fresh,incremental;
        check(conversation_snapshot_save(fresh,view,ss,g,draft.state,err),"full capture reference after growth or rewind");
        size_t peak=0,reused=0;
        check(conversation_snapshot_capture_bytes(reuse,view,ss,g,draft.state,peak,err),"admit incremental capture peak");
        check(conversation_snapshot_save(incremental,view,ss,g,draft.state,err,std::move(reuse),&reused),"capture with retained pages");
        check(incremental.bytes() <= peak,"incremental allocation stays within admitted bound");
        check(incremental.live.gdn==fresh.live.gdn && incremental.live.ple==fresh.live.ple &&
              incremental.live.dead==fresh.live.dead && equal(incremental.kv[0],fresh.kv[0]) &&
              equal(incremental.kv[1],fresh.kv[1]),"incremental capture equals full capture after growth or rewind");
        check((dirty>=4)==(reused>0),"only complete unchanged pages or rows are retained");
        incremental.kv[0].k.visit(0,1,[&](const uint8_t* p,size_t,size_t){
            check(p==original,"growth never reallocates the retained payload");return true;
        });
        check(conversation_snapshot_restore(incremental,ss,g,draft.state,err)==ConversationRestore::restored,"restore segmented incremental snapshot");
        uint64_t fingerprint=0;
        check(conversation_kv_verify(incremental.kv.back(),draft.state,g,70,false,fingerprint,err),"incremental draft authoritative and ring read-back");
        ids.resize(65);
    }
    {
        // a layer split's later stage: the same image WITHOUT the draft layer's K/V (draft == nullptr)
        check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore A before stage images");
        size_t with_draft=0,without=0;
        check(conversation_snapshot_bytes(view,ss,g,draft.state,with_draft,err) &&
              conversation_snapshot_bytes(view,ss,g,nullptr,without,err) && without<with_draft,
              "a stage image's estimate leaves the draft out");
        SavedConversation stage,stage_b,stage_back;
        check(conversation_snapshot_save(stage,view,ss,g,nullptr,err),"capture a stage image");
        check(stage.kv.size()==a.kv.size()-1 && equal(stage.kv[0],a.kv[0]) && stage.live.gdn==a.live.gdn,
              "a stage image holds the session's own layers, no draft");
        check(stage.bytes()<a.bytes(),"a stage image is smaller by the draft ring");
        check(conversation_snapshot_validate(stage,ss,g,nullptr,err),"validate a stage image without a draft");
        check(!conversation_snapshot_validate(stage,ss,g,draft.state,err),"a stage image is refused where a draft is expected");
        check(!conversation_snapshot_validate(a,ss,g,nullptr,err),"an image with a draft is refused where none is expected");
        fill(201);
        check(conversation_snapshot_save(stage_b,view,ss,g,nullptr,err),"capture another stage state");
        check(conversation_snapshot_restore(a,ss,g,nullptr,err)==ConversationRestore::invalid,
              "restoring a draft image as a stage image is refused before any write");
        check(conversation_snapshot_save(stage_back,view,ss,g,nullptr,err) && stage_back.live.gdn==stage_b.live.gdn &&
              equal(stage_back.kv[0],stage_b.kv[0]),"the refusal left the stage untouched");
        check(conversation_snapshot_restore(stage,ss,g,nullptr,err)==ConversationRestore::restored,"restore a stage image");
        check(conversation_snapshot_save(stage_back,view,ss,g,nullptr,err) && stage_back.live.gdn==stage.live.gdn &&
              stage_back.live.dead==stage.live.dead && equal(stage_back.kv[0],stage.kv[0]),"stage A/B/A exactness");
        // its retained K/V: the next park copies only what changed, and equals a full capture
        ConversationKvReuse reuse{stage.kv,65,65,{}};
        SavedConversation fresh,incremental;
        size_t peak=0,reused=0;
        check(conversation_snapshot_save(fresh,view,ss,g,nullptr,err),"full stage capture reference");
        check(conversation_snapshot_capture_bytes(reuse,view,ss,g,nullptr,peak,err),"admit an incremental stage capture");
        check(conversation_snapshot_save(incremental,view,ss,g,nullptr,err,std::move(reuse),&reused),"incremental stage capture");
        check(reused>0 && incremental.bytes()<=peak && equal(incremental.kv[0],fresh.kv[0]) &&
              incremental.live.gdn==fresh.live.gdn,"an incremental stage capture reuses pages and equals a full one");
        SavedConversation split_image=stage;
        split_image.stage_images.push_back(stage);
        check(!conversation_snapshot_validate(split_image,ss,g,draft.state,err),
              "a layer split's image is refused by the whole-session form");
        ConversationKvReuse wrong{a.kv,65,65,{}};
        check(!conversation_snapshot_capture_bytes(wrong,view,ss,g,nullptr,peak,err),"a draft image's K/V is not a stage's reuse");
    }
    check(conversation_snapshot_restore(a,ss,g,draft.state,err)==ConversationRestore::restored,"restore A after stage images");
    check(conversation_checkpoint_restore(a.checkpoints[0],ss,g,err),"restore early running checkpoint");
    std::vector<uint8_t> spare(sizes.dead);
    cuda_check(cudaMemcpy(spare.data(),main.state.idx_pooled,sizes.dead,cudaMemcpyDeviceToHost));
    check(spare==a.checkpoints[0].dead,"checkpoint rebuilds spare row over a later completed block");
    check(ss.ple_prev[0]==2 && ss.ple_prev[1]==3,"checkpoint PLE token window");
}
}

int main() {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    for (int fmt : std::array<int,4>{kKvF16,kKvInt8,kKvQ4,3}) for (int mode : {0,1,2}) {
        if (fmt==3 && mode!=0) continue;
        Fixture f(fmt,mode);
        if (fmt==3) {
            std::string err;
            ConversationKv invalid;
            for (int unsupported : {1,2}) {
                f.state.kv_mode=unsupported;
                check(conversation_kv_bytes(f.state,f.g,9,true)==0,"hybrid streaming/ring rejected without block-mover abort");
                check(!conversation_kv_save(invalid,f.state,f.g,9,true,err),"hybrid streaming/ring capture fails closed");
            }
            f.state.kv_mode=0;
        }
        {
            std::string err;
            ConversationKv invalid;
            for (const int64_t bad : {-1LL, 97LL, (long long) std::numeric_limits<int64_t>::max()}) {
                check(!conversation_kv_save(invalid,f.state,f.g,bad,mode!=2,err),"reject invalid save extent before allocation");
                check(!conversation_kv_restore(invalid,f.state,f.g,bad,mode!=2,err),"reject invalid restore extent before copying");
                check(conversation_kv_bytes(f.state,f.g,bad,mode!=2)==0,"invalid extent cannot overflow byte estimate");
            }
        }
        for (int64_t upto : {0,1,3,4,5,63,64,65,96}) {
            const bool index=mode!=2;
            std::string err;
            f.fill(13);
            ConversationKv a,b,restored;
            cuda_check(cudaDeviceSynchronize());
            check(conversation_kv_save(a,f.state,f.g,upto,index,err),"save A");
            check(a.bytes()==conversation_kv_bytes(f.state,f.g,upto,index),"size estimate equals snapshot payload");
            f.fill(177);
            check(conversation_kv_save(b,f.state,f.g,upto,index,err),"save B");
            check(upto==0 || !equal(a,b),"fixture changes state");
            check(conversation_kv_restore(a,f.state,f.g,upto,index,err),"restore A over B");
            cuda_check(cudaDeviceSynchronize());
            check(conversation_kv_save(restored,f.state,f.g,upto,index,err),"read back restored A");
            check(equal(a,restored),"A/B/A byte-exact K/V and indexer state");
            uint64_t fingerprint = 0;
            check(conversation_kv_verify(a,f.state,f.g,upto,index,fingerprint,err),"verify restored authoritative and resident bytes without rewriting them");
            {
                // disk save streams the same bytes as a captured image, without the host copy
                SessionKvSource src;
                check(conversation_kv_source(src,f.state,f.g,upto,index,err),"K/V source for a disk save");
                check(src.format==a.format && src.cells==a.cells && src.heads==a.heads && src.head_dim==a.head_dim &&
                      src.page_size==a.page_size && src.pooled_rows==a.pooled_rows && src.idx_dim==a.idx_dim,
                      "K/V source metadata equals the captured image");
                const std::array<const ConversationBuffer*,5> parts={&a.k,&a.v,&a.k_scale,&a.v_scale,&a.pooled};
                for (size_t i=0;i<5;++i) {
                    check(src.sizes[i]==parts[i]->size(),"K/V source part size");
                    std::vector<uint8_t> got(src.sizes[i]), want(src.sizes[i]);
                    const size_t half=src.sizes[i]/2;   // two reads: offsets are honoured
                    check(src.read(i,0,got.data(),half) && src.read(i,half,got.data()+half,src.sizes[i]-half),
                          "K/V source reads");
                    check(parts[i]->read(want.data(),0,want.size()) && got==want,"K/V source bytes equal the image");
                }
                SessionKvSource none;
                check(!conversation_kv_source(none,f.state,f.g,-1,index,err),"K/V source rejects an invalid extent");
            }
            if (mode==1) {
                std::vector<int32_t> table((size_t)f.state.n_pages);
                cuda_check(cudaMemcpy(table.data(),f.state.page_table,table.size()*4,cudaMemcpyDeviceToHost));
                check(std::all_of(table.begin(),table.end(),[](int32_t x){return x==-1;}),"stale VRAM pages invalidated");
            }
            if (mode==2 && upto>0) {
                const auto s=qsa_shapes(f.g);
                const int64_t b1=(upto+s.page_size-1)/s.page_size,b0=std::max<int64_t>(0,b1-f.state.n_slots);
                const size_t block=kv_block_bytes(s,fmt)/2;
                // Codes and scales are separate in INT8; check code bytes here.
                const size_t code_block=fmt==kKvInt8 ? s.page_size*s.n_head_kv*s.head_dim : block;
                const void* slots=fmt==kKvQ4?(void*)f.state.k_q4:fmt==kKvInt8?(void*)f.state.k_q:(void*)f.state.k_pool;
                std::vector<uint8_t> got(code_block);
                for (int64_t page=b0;page<b1;++page) {
                    cuda_check(cudaMemcpy(got.data(),(const uint8_t*)slots+(page%f.state.n_slots)*code_block,code_block,cudaMemcpyDeviceToHost));
                    std::vector<uint8_t> expected(code_block);
                    check(a.k.read(expected.data(), page*code_block, code_block),"read segmented expected draft page");
                    check(got == expected,"draft ring contains restored pages");
                }
            }
            auto bad=a; bad.head_dim++;
            check(!conversation_kv_restore(bad,f.state,f.g,upto,index,err),"reject incompatible geometry");
            if (fmt==kKvInt8) {
                // #293: INT8 K/V saved through the Hadamard rotation (STRATA_KV_ROT=1) never restores into a state
                // without it, nor the reverse
                f.state.kv_rot=true;
                ConversationKv rotated;
                check(conversation_kv_save(rotated,f.state,f.g,upto,index,err),"save rotated INT8");
                check(!conversation_kv_restore(a,f.state,f.g,upto,index,err),"unrotated image into a rotated state rejected");
                f.state.kv_rot=false;
                check(!conversation_kv_restore(rotated,f.state,f.g,upto,index,err),"rotated image into an unrotated state rejected");
                check(conversation_kv_restore(a,f.state,f.g,upto,index,err),"unrotated image still restores");
            }
            if (!a.k.empty()) {
                bad=a; bad.k.pop_back();
                check(!conversation_kv_restore(bad,f.state,f.g,upto,index,err),"reject malformed payload");
            }
        }
    }
    for (int fmt : {kKvF16,kKvInt8,kKvQ4}) for (int mode : {0,1}) for (int experts : {256,512})
        full_session(fmt,mode,experts);
    for (int experts : {256,512}) full_session(3,0,experts);
    for (int fmt : std::array<int,4>{kKvF16,kKvInt8,kKvQ4,3}) pooled_snapshots(fmt);
    pooled_segment_crossing();
    std::printf("conversation_snapshot_test: %d checks passed\n",checks);
}
