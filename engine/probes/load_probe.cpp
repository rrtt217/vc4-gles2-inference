// How long does the 69 MB q8 checkpoint take to reach RAM on this board?
// Methods: buffered ifstream (what the engine does today), mmap + memcpy into
// the model vectors, mmap alone (borrowed bytes, no copy), 1 MiB read loop,
// and mmap + MADV_WILLNEED + copy.
// Usage: load_probe [model] [start-method 0..4] [cold]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
using Clock = std::chrono::steady_clock;
static const int kMethods = 5;
static bool g_cold = false;
static const char* g_path = nullptr;
static double ms(Clock::time_point a){ return std::chrono::duration<double, std::milli>(Clock::now() - a).count(); }
static size_t n_of(const char* p){ struct stat st {}; if (stat(p, &st)) { perror("stat"); exit(1); } return (size_t)st.st_size; }
static void drop_cache(){
    // Clean page-cache pages of one file can be dropped without root, which
    // gives a true cold read per method instead of a warm one.
    int fd = open(g_path, O_RDONLY);
    if (fd >= 0) { posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); close(fd); }
}
static void m_ifstream(size_t n){
    auto t0 = Clock::now(); std::ifstream f(g_path, std::ios::binary);
    std::vector<int8_t> v(n); f.read((char*)v.data(), (std::streamsize)n);
    printf("ifstream    %8.1f ms  %zu B ok=%d\n", ms(t0), n, (bool)f);
}
static void m_mmap_copy(size_t n){
    auto t0 = Clock::now(); int fd = open(g_path, O_RDONLY);
    const uint8_t* b = (const uint8_t*)mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    std::vector<int8_t> v(n); memcpy(v.data(), b, n); munmap((void*)b, n);
    printf("mmap+copy   %8.1f ms\n", ms(t0));
}
static void m_mmap_touch(size_t n){
    auto t0 = Clock::now(); int fd = open(g_path, O_RDONLY);
    const uint8_t* b = (const uint8_t*)mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    size_t s = 0; for (size_t i = 0; i < n; i += 4096) s += b[i]; munmap((void*)b, n);
    printf("mmap+touch  %8.1f ms  (sum %zu)\n", ms(t0), s);
}
static void m_read(size_t n){
    auto t0 = Clock::now(); int fd = open(g_path, O_RDONLY);
    std::vector<uint8_t> buf(1u << 20); size_t got = 0; ssize_t r;
    while ((r = read(fd, buf.data(), buf.size())) > 0) got += (size_t)r; close(fd);
    printf("read 1MiB   %8.1f ms  (%zu B)\n", ms(t0), got);
}
static void m_mmap_willneed(size_t n){
    auto t0 = Clock::now(); int fd = open(g_path, O_RDONLY);
    const uint8_t* b = (const uint8_t*)mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
    madvise((void*)b, n, MADV_WILLNEED);
    std::vector<int8_t> v(n); memcpy(v.data(), b, n); munmap((void*)b, n);
    printf("mmap+will   %8.1f ms\n", ms(t0));
}
int main(int argc, char** argv){
    g_path = argc > 1 ? argv[1] : "../models/minimind3/minimind3.q8";
    const size_t n = n_of(g_path);
    int start = argc > 2 ? atoi(argv[2]) : 0;
    g_cold = argc > 3 && !strcmp(argv[3], "cold");
    for (int k = 0; k < kMethods; ++k) {
        if (g_cold) drop_cache();
        switch ((k + start) % kMethods) {
        case 0: m_ifstream(n); break;
        case 1: m_mmap_copy(n); break;
        case 2: m_mmap_touch(n); break;
        case 3: m_read(n); break;
        case 4: m_mmap_willneed(n); break;
        }
    }
    return 0;
}
