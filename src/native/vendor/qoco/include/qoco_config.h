#ifndef QOCO_CONFIG_H
#define QOCO_CONFIG_H

#define QOCO_VERSION_MAJOR 0
#define QOCO_VERSION_MINOR 3
#define QOCO_VERSION_PATCH 2
#define QOCO_VERSION "0.3.2"

#if defined(_WIN32)
#define IS_WINDOWS
#elif defined(__APPLE__)
#define IS_MACOS
#else
#define IS_LINUX
#endif

#endif
