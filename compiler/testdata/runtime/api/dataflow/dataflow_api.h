// Stand-in declarations of the tt-metal dataflow calls the Tessera device
// runtime uses, for a syntax check on a machine without tt-metal.
#pragma once
#include <cstdint>
template <typename T> T get_arg_val(uint32_t);
uint32_t get_write_ptr(uint32_t);
uint64_t get_noc_addr(uint32_t x, uint32_t y, uint32_t addr);
void noc_async_write(uint32_t, uint64_t, uint32_t);
void noc_async_write_barrier();
void noc_async_read(uint64_t, uint32_t, uint32_t);
void noc_async_read_barrier();
template <bool DRAM> struct InterleavedAddrGen {
    uint32_t bank_base_address;
    uint32_t page_size;
    uint64_t get_noc_addr(uint32_t id, uint32_t offset = 0) const;
};
void cb_reserve_back(uint32_t, uint32_t);
void cb_push_back(uint32_t, uint32_t);
void cb_wait_front(uint32_t, uint32_t);
void cb_pop_front(uint32_t, uint32_t);
uint32_t get_read_ptr(uint32_t);
void noc_semaphore_inc(uint64_t, uint32_t);
void noc_async_atomic_barrier();
