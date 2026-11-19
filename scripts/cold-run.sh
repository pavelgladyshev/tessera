#!/usr/bin/env bash
# Cold-cache comparison: evict the image from the page cache before each run.
set -u
cd ~/tessera
export TT_METAL_HOME=~/git/tt-metal-v0.80.0 TT_METAL_RUNTIME_ROOT=~/git/tt-metal-v0.80.0 TT_METAL_CACHE=~/tessera/.jit-cache
(cd deca-full && ./build.sh 2>&1 | grep -E "error|built")
evict() { python3 -c "import os,sys; fd=os.open(sys.argv[1], os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)" "$1"; }
run() {  # name dir command...
    local name=$1 dir=$2; shift 2
    rm -rf "$dir" && mkdir -p "$dir" && ln -sf "$IMG" "$dir/disk.img" && ln -sf ~/tessera/models/ndeca4096.tsw "$dir/ndeca4096.tsw"
    evict "$IMG"
    (cd "$dir" && /usr/bin/time -f "TIME %e %I" "$@" > out.txt 2> err.txt)
    local t=$(grep "^TIME" "$dir/err.txt" | awk '{print $2}') io=$(grep "^TIME" "$dir/err.txt" | awk '{printf "%.1f", $3*512/1e6}')
    local workers=$(grep -o "workers ran in [0-9.]* ms" "$dir/err.txt" | grep -o "[0-9.]* ms")
    local cdir="$dir/carved"; [ -d "$dir/out" ] && cdir="$dir/out"
    (cd "$cdir" 2>/dev/null && sha1sum * | sort -k2) > "$dir.sha1"
    printf "  %-34s %6s s   disk read %8s MB   workers %10s   files %3d\n" "$name" "$t" "$io" "${workers:--}" "$(wc -l < "$dir.sha1")"
}
for img in case10_zeros case2_zeros; do
    IMG=~/ndeca-data/paper/images/$img.raw
    echo "$img ($(du -m $IMG | cut -f1) MB):"
    run "2017 C carver (--deca)" cold/$img-2017 sh -c 'mkdir -p out && ../../deca2017/deca2017 --deca -o out disk.img'
    run "Tessera DECA, whole image to card" cold/$img-full ../../deca-full/build/src_deca_full --img disk.img
    run "Tessera DECA, on demand" cold/$img-od ../../deca/build/deca --img disk.img
    run "Tessera DECA + ndeca, on demand" cold/$img-ndeca ../../deca-ndeca/build/deca_ndeca --img disk.img --det ndeca4096.tsw
    for v in full od; do printf "    %-6s same files as before: %s\n" $v "$(cmp -s cold/$img-$v.sha1 deca-card/$img.sha1 && echo yes || echo NO)"; done
    printf "    ndeca  same files as before: %s\n" "$(cmp -s cold/$img-ndeca.sha1 deca-ndeca-card/$img.sha1 && echo yes || echo NO)"
    printf "    2017   same files as before: %s\n" "$(cmp -s cold/$img-2017.sha1 deca2017/out-$img.sha1 && echo yes || echo NO)"
    grep -h "carving read\|served" cold/$img-od/err.txt cold/$img-ndeca/err.txt | sed "s/^/    /"
done
echo "card holders: $(lsof /dev/tenstorrent/* 2>/dev/null | wc -l)"
