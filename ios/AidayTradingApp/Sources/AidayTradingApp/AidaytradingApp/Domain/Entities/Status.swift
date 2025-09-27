import Foundation

struct Status: Codable, Equatable {
    let running: Bool
    let uptimeSeconds: Int

    var formattedUptime: String {
        guard uptimeSeconds > 0 else { return "" }
        let hours = uptimeSeconds / 3600
        let minutes = (uptimeSeconds % 3600) / 60
        if hours > 0 {
            return "Uptime: \(hours)h \(minutes)m"
        } else {
            return "Uptime: \(minutes)m"
        }
    }
}
