/*
 * A stand-in for The Sleuth Kit, just enough to build the 2017 DECA carver
 * unchanged in disk mode (one raw image file). Volume and file-system access
 * are stubs that fail, so -p, -u and -e are not available.
 */
#ifndef TSK_SHIM_LIBTSK_H
#define TSK_SHIM_LIBTSK_H
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

typedef char TSK_TCHAR;
typedef uint64_t TSK_DADDR_T;
typedef uint32_t TSK_PNUM_T;
#define PRIuPNUM PRIu32
#define PRIuDADDR PRIu64

typedef enum { TSK_IMG_TYPE_DETECT = 0 } TSK_IMG_TYPE_ENUM;
typedef enum { TSK_VS_TYPE_DETECT = 0 } TSK_VS_TYPE_ENUM;
typedef enum { TSK_FS_TYPE_DETECT = 0 } TSK_FS_TYPE_ENUM;
typedef enum { TSK_WALK_CONT = 0, TSK_WALK_STOP = 1, TSK_WALK_ERROR = 2 } TSK_WALK_RET_ENUM;
typedef enum { TSK_VS_PART_FLAG_ALLOC = 1, TSK_VS_PART_FLAG_UNALLOC = 2, TSK_VS_PART_FLAG_META = 4,
               TSK_VS_PART_FLAG_ALL = 7 } TSK_VS_PART_FLAG_ENUM;
typedef enum { TSK_FS_BLOCK_WALK_FLAG_ALLOC = 1, TSK_FS_BLOCK_WALK_FLAG_UNALLOC = 2,
               TSK_FS_BLOCK_WALK_FLAG_AONLY = 0x40 } TSK_FS_BLOCK_WALK_FLAG_ENUM;

typedef struct { int fd; unsigned int sector_size; int64_t size; } TSK_IMG_INFO;
typedef struct { unsigned int block_size; TSK_PNUM_T part_count; } TSK_VS_INFO;
typedef struct {
    TSK_DADDR_T start, len;
    TSK_PNUM_T addr;
    int8_t table_num, slot_num;
    int flags;
    const char *desc;
} TSK_VS_PART_INFO;
typedef struct { unsigned int block_size; TSK_DADDR_T block_count, first_block, last_block; } TSK_FS_INFO;
typedef struct { TSK_DADDR_T addr; } TSK_FS_BLOCK;
typedef TSK_WALK_RET_ENUM (*TSK_FS_BLOCK_WALK_CB)(TSK_FS_BLOCK *, void *);
typedef TSK_WALK_RET_ENUM (*TSK_VS_PART_WALK_CB)(TSK_VS_INFO *, const TSK_VS_PART_INFO *, void *);

TSK_IMG_INFO *tsk_img_open(int n, const TSK_TCHAR *const *paths, TSK_IMG_TYPE_ENUM type, unsigned int ssize);
ssize_t tsk_img_read(TSK_IMG_INFO *img, int64_t off, char *buf, size_t len);
void tsk_img_close(TSK_IMG_INFO *img);

TSK_VS_INFO *tsk_vs_open(TSK_IMG_INFO *img, TSK_DADDR_T off, TSK_VS_TYPE_ENUM type);
const TSK_VS_PART_INFO *tsk_vs_part_get(const TSK_VS_INFO *vs, TSK_PNUM_T idx);
ssize_t tsk_vs_part_read_block(const TSK_VS_PART_INFO *part, TSK_DADDR_T addr, char *buf, size_t len);
uint8_t tsk_vs_part_walk(TSK_VS_INFO *vs, TSK_PNUM_T start, TSK_PNUM_T last, int flags, TSK_VS_PART_WALK_CB cb, void *ptr);
void tsk_vs_close(TSK_VS_INFO *vs);

TSK_FS_INFO *tsk_fs_open_vol(const TSK_VS_PART_INFO *part, TSK_FS_TYPE_ENUM type);
ssize_t tsk_fs_read_block(TSK_FS_INFO *fs, TSK_DADDR_T addr, char *buf, size_t len);
uint8_t tsk_fs_block_walk(TSK_FS_INFO *fs, TSK_DADDR_T start, TSK_DADDR_T end, int flags, TSK_FS_BLOCK_WALK_CB cb, void *ptr);
void tsk_fs_close(TSK_FS_INFO *fs);
#endif
