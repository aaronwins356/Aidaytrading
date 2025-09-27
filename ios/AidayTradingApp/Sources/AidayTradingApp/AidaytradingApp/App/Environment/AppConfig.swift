import Foundation

struct AppConfig {
    static let baseURL: URL = {
        #if DEBUG
        URL(string: "https://dev.api.yourdomain.com")!
        #elseif STAGING
        URL(string: "https://staging.api.yourdomain.com")!
        #else
        URL(string: "https://api.yourdomain.com")!
        #endif
    }()

    static let webSocketURL: URL = {
        #if DEBUG
        URL(string: "wss://api.example.com/ws")!
        #elseif STAGING
        URL(string: "wss://api.example.com/ws")!
        #else
        URL(string: "wss://api.example.com/ws")!
        #endif
    }()

    static let pollingInterval: TimeInterval = 600
    static let centralTimeZone: TimeZone = TimeZone(identifier: "America/Chicago")!
}
