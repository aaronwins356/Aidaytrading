import Foundation

enum RoleManager {
    static func isAdmin(_ profile: UserProfile) -> Bool {
        profile.role == .admin
    }

    static func canApproveUsers(_ profile: UserProfile) -> Bool {
        isAdmin(profile)
    }

    static func accessibleTabs(for profile: UserProfile) -> [MainTabView.Tab] {
        if isAdmin(profile) {
            return [.home, .calendar, .trades, .admin]
        }
        return [.home, .calendar, .trades]
    }
}
