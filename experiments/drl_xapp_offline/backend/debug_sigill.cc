// Diagnostic-only preload; never used by normal experiments.
#define _GNU_SOURCE
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <execinfo.h>
#include <fcntl.h>
#include <unistd.h>
#include <ucontext.h>

static void Failure(int signal, siginfo_t* info, void* context)
{
    auto* ctx = static_cast<ucontext_t*>(context);
    dprintf(STDERR_FILENO, "\nDIAGNOSTIC_SIGNAL signal=%d si_code=%d fault=%p rip=%p\n",
            signal, info->si_code, info->si_addr,
            reinterpret_cast<void*>(ctx->uc_mcontext.gregs[REG_RIP]));
    void* stack[80];
    int count = backtrace(stack, 80);
    backtrace_symbols_fd(stack, count, STDERR_FILENO);
    dprintf(STDERR_FILENO, "\nDIAGNOSTIC_MAPS_BEGIN\n");
    int maps = open("/proc/self/maps", O_RDONLY);
    char buffer[4096];
    ssize_t bytes;
    while (maps >= 0 && (bytes = read(maps, buffer, sizeof(buffer))) > 0)
        write(STDERR_FILENO, buffer, bytes);
    if (maps >= 0) close(maps);
    dprintf(STDERR_FILENO, "\nDIAGNOSTIC_MAPS_END\n");
    _exit(128 + signal);
}

__attribute__((constructor)) static void Install()
{
    struct sigaction action{};
    action.sa_sigaction = Failure;
    action.sa_flags = SA_SIGINFO | SA_RESETHAND;
    sigemptyset(&action.sa_mask);
    sigaction(SIGILL, &action, nullptr);
    sigaction(SIGSEGV, &action, nullptr);
    sigaction(SIGABRT, &action, nullptr);
}
