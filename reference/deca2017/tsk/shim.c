/* Raw-image implementation of the libtsk calls the 2017 DECA carver makes. */
#define _FILE_OFFSET_BITS 64
#include <fcntl.h>
#include <sys/stat.h>
#include "libtsk.h"

TSK_IMG_INFO *tsk_img_open(int n, const TSK_TCHAR *const *paths, TSK_IMG_TYPE_ENUM type, unsigned int ssize)
{
    (void)type;
    if (n != 1) {
        fprintf(stderr, "tsk shim: one raw image file only\n");
        return NULL;
    }
    int fd = open(paths[0], O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return NULL; }
    TSK_IMG_INFO *img = calloc(1, sizeof *img);
    img->fd = fd;
    img->sector_size = ssize ? ssize : 512;
    img->size = st.st_size;
    return img;
}

/* Like libtsk: a read that starts at or past the end fails and leaves buf alone. */
ssize_t tsk_img_read(TSK_IMG_INFO *img, int64_t off, char *buf, size_t len)
{
    if (off < 0 || off >= img->size) return -1;
    return pread(img->fd, buf, len, off);
}

void tsk_img_close(TSK_IMG_INFO *img)
{
    if (img) { close(img->fd); free(img); }
}

TSK_VS_INFO *tsk_vs_open(TSK_IMG_INFO *img, TSK_DADDR_T off, TSK_VS_TYPE_ENUM type)
{
    (void)img; (void)off; (void)type;
    fprintf(stderr, "tsk shim: volumes are not supported\n");
    return NULL;
}
const TSK_VS_PART_INFO *tsk_vs_part_get(const TSK_VS_INFO *vs, TSK_PNUM_T idx) { (void)vs; (void)idx; return NULL; }
ssize_t tsk_vs_part_read_block(const TSK_VS_PART_INFO *part, TSK_DADDR_T addr, char *buf, size_t len)
{
    (void)part; (void)addr; (void)buf; (void)len;
    return -1;
}
uint8_t tsk_vs_part_walk(TSK_VS_INFO *vs, TSK_PNUM_T start, TSK_PNUM_T last, int flags, TSK_VS_PART_WALK_CB cb, void *ptr)
{
    (void)vs; (void)start; (void)last; (void)flags; (void)cb; (void)ptr;
    return 1;
}
void tsk_vs_close(TSK_VS_INFO *vs) { (void)vs; }
TSK_FS_INFO *tsk_fs_open_vol(const TSK_VS_PART_INFO *part, TSK_FS_TYPE_ENUM type) { (void)part; (void)type; return NULL; }
ssize_t tsk_fs_read_block(TSK_FS_INFO *fs, TSK_DADDR_T addr, char *buf, size_t len)
{
    (void)fs; (void)addr; (void)buf; (void)len;
    return -1;
}
uint8_t tsk_fs_block_walk(TSK_FS_INFO *fs, TSK_DADDR_T start, TSK_DADDR_T end, int flags, TSK_FS_BLOCK_WALK_CB cb, void *ptr)
{
    (void)fs; (void)start; (void)end; (void)flags; (void)cb; (void)ptr;
    return 1;
}
void tsk_fs_close(TSK_FS_INFO *fs) { (void)fs; }
