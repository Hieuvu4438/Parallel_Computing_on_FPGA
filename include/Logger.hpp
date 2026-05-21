/**
 * @file Logger.hpp
 * @brief Logging utility for Respiratory Sound Analysis System
 *
 * Cung cấp logging đơn giản với các mức độ khác nhau.
 * Thiết kế nhẹ, không phụ thuộc thư viện ngoài.
 */

#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>

namespace respiratory {

// ============================================================================
// LOG LEVELS
// ============================================================================

enum class LogLevel : int {
    LVL_DEBUG   = 0,   ///< Debug - chi tiết nhất
    LVL_INFO    = 1,   ///< Info  - thông tin chung
    LVL_WARNING = 2,   ///< Warning - cảnh báo
    LVL_ERROR   = 3,   ///< Error - lỗi nghiêm trọng
    LVL_NONE    = 4    ///< Tắt log
};

// ============================================================================
// LOGGER CLASS
// ============================================================================

/**
 * @class Logger
 * @brief Simple thread-unsafe logger (sufficient for single-threaded pipeline)
 */
class Logger {
public:
    /**
     * @brief Lấy instance singleton
     */
    static Logger& instance() {
        static Logger inst;
        return inst;
    }

    /**
     * @brief Đặt mức độ log tối thiểu
     */
    void setLevel(LogLevel level) { m_level = level; }

    /**
     * @brief Lấy mức độ log hiện tại
     */
    LogLevel getLevel() const { return m_level; }

    /**
     * @brief Bật/tắt timestamp
     */
    void setShowTimestamp(bool show) { m_showTimestamp = show; }

    /**
     * @brief Ghi một dòng log
     */
    void log(LogLevel level, const std::string& module, const std::string& message) {
        if (level < m_level) return;

        std::ostream& out = (level >= LogLevel::LVL_WARNING) ? std::cerr : std::cout;

        if (m_showTimestamp) {
            out << "[" << getTimestamp() << "] ";
        }

        out << levelPrefix(level)
            << "[" << module << "] "
            << message << "\n";
    }

    // Convenience wrappers
    void debug(const std::string& module, const std::string& msg) {
        log(LogLevel::LVL_DEBUG, module, msg);
    }
    void info(const std::string& module, const std::string& msg) {
        log(LogLevel::LVL_INFO, module, msg);
    }
    void warning(const std::string& module, const std::string& msg) {
        log(LogLevel::LVL_WARNING, module, msg);
    }
    void error(const std::string& module, const std::string& msg) {
        log(LogLevel::LVL_ERROR, module, msg);
    }

private:
    Logger() : m_level(LogLevel::LVL_INFO), m_showTimestamp(false) {}
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    LogLevel m_level;
    bool m_showTimestamp;

    static const char* levelPrefix(LogLevel level) {
        switch (level) {
            case LogLevel::LVL_DEBUG:   return "[DEBUG]   ";
            case LogLevel::LVL_INFO:    return "[INFO]    ";
            case LogLevel::LVL_WARNING: return "[WARNING] ";
            case LogLevel::LVL_ERROR:   return "[ERROR]   ";
            default:                    return "[?]       ";
        }
    }

    static std::string getTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto t   = std::chrono::system_clock::to_time_t(now);
        std::ostringstream oss;
        oss << std::put_time(std::localtime(&t), "%H:%M:%S");
        return oss.str();
    }
};

// ============================================================================
// CONVENIENCE MACROS
// ============================================================================

#define LOG_DEBUG(module, msg)   respiratory::Logger::instance().debug(module, msg)
#define LOG_INFO(module, msg)    respiratory::Logger::instance().info(module, msg)
#define LOG_WARNING(module, msg) respiratory::Logger::instance().warning(module, msg)
#define LOG_ERROR(module, msg)   respiratory::Logger::instance().error(module, msg)

} // namespace respiratory

#endif // LOGGER_HPP
