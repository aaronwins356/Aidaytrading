import Foundation

struct NotificationPreferences: Codable, Equatable {
    var botEventsEnabled: Bool
    var reportsEnabled: Bool

    static let `default` = NotificationPreferences(botEventsEnabled: true, reportsEnabled: true)
}
