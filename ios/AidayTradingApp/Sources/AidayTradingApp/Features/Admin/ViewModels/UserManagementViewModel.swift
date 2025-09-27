import Combine
import Foundation

@MainActor
final class UserManagementViewModel: ObservableObject {
    @Published private(set) var users: [AdminUser] = []
    @Published private(set) var isLoading = false
    @Published private(set) var isPerformingAction = false
    @Published var errorMessage: String?
    @Published var bannerMessage: String?

    weak var actionRecorder: AdminActionRecording?

    private let repository: AdminRepository

    init(repository: AdminRepository) {
        self.repository = repository
    }

    func loadUsers() async {
        isLoading = true
        defer { isLoading = false }
        do {
            users = try await repository.fetchUsers().sorted { self.userSort(lhs: $0, rhs: $1) }
        } catch {
            errorMessage = "Failed to load users: \(error.localizedDescription)"
        }
    }

    func approve(_ user: AdminUser) async {
        await update(user: user, newStatus: .active, newRole: .viewer, successMessage: "Approved \(user.username)")
    }

    func disable(_ user: AdminUser) async {
        await update(user: user, newStatus: .disabled, newRole: nil, successMessage: "Disabled \(user.username)")
    }

    func activate(_ user: AdminUser) async {
        await update(user: user, newStatus: .active, newRole: user.role, successMessage: "Activated \(user.username)")
    }

    func setRole(_ user: AdminUser, role: AdminUser.Role) async {
        guard user.role != role else { return }
        await update(user: user, newStatus: nil, newRole: role, successMessage: "Updated \(user.username) role")
    }

    func resetPassword(_ user: AdminUser) async {
        guard !isPerformingAction else { return }
        isPerformingAction = true
        bannerMessage = nil
        defer { isPerformingAction = false }
        do {
            try await repository.resetPassword(id: user.id)
            bannerMessage = "Password reset email sent to \(user.email)"
            actionRecorder?.recordChange(
                category: .user,
                summary: "Password reset issued",
                details: "Reset link sent for \(user.username)",
                payload: ["user_id": user.id.uuidString, "action": "reset_password"],
                showBanner: false
            )
        } catch {
            errorMessage = "Failed to trigger password reset: \(error.localizedDescription)"
        }
    }

    private func update(
        user: AdminUser,
        newStatus: AdminUser.Status?,
        newRole: AdminUser.Role?,
        successMessage: String
    ) async {
        guard !isPerformingAction else { return }
        isPerformingAction = true
        bannerMessage = nil
        defer { isPerformingAction = false }
        do {
            let updated = try await repository.updateUser(id: user.id, role: newRole, status: newStatus)
            users = users
                .map { $0.id == updated.id ? updated : $0 }
                .sorted { self.userSort(lhs: $0, rhs: $1) }
            bannerMessage = successMessage
            var details = "Status: \(updated.status.displayName), Role: \(updated.role.displayName)"
            if updated.status == .pending { details.append(" (pending)") }
            actionRecorder?.recordChange(
                category: .user,
                summary: successMessage,
                details: details,
                payload: [
                    "user_id": updated.id.uuidString,
                    "role": updated.role.rawValue,
                    "status": updated.status.rawValue
                ],
                showBanner: true
            )
        } catch {
            errorMessage = "Failed to update user: \(error.localizedDescription)"
        }
    }

    private func userSort(lhs: AdminUser, rhs: AdminUser) -> Bool {
        if lhs.status == rhs.status {
            return lhs.username.lowercased() < rhs.username.lowercased()
        }
        if lhs.status == .pending { return true }
        if rhs.status == .pending { return false }
        return lhs.username.lowercased() < rhs.username.lowercased()
    }
}
