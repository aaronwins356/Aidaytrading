import SwiftUI

struct UserDetailView: View {
    @ObservedObject var viewModel: UserManagementViewModel
    let user: AdminUser
    let onDismiss: () -> Void

    @State private var selectedRole: AdminUser.Role

    init(viewModel: UserManagementViewModel, user: AdminUser, onDismiss: @escaping () -> Void) {
        self.viewModel = viewModel
        self.user = user
        self.onDismiss = onDismiss
        _selectedRole = State(initialValue: user.role)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section(header: Text("Account")) {
                    HStack {
                        Text("Username")
                        Spacer()
                        Text(currentUser.username)
                            .foregroundColor(.secondary)
                    }
                    HStack {
                        Text("Email")
                        Spacer()
                        Text(currentUser.email)
                            .foregroundColor(.secondary)
                    }
                    HStack {
                        Text("Status")
                        Spacer()
                        Text(currentUser.status.displayName)
                            .foregroundColor(statusColor)
                    }
                }

                Section(header: Text("Role")) {
                    Picker("Role", selection: $selectedRole) {
                        ForEach(AdminUser.Role.allCases) { role in
                            Text(role.displayName).tag(role)
                        }
                    }
                    .pickerStyle(.segmented)
                    .disabled(viewModel.isPerformingAction)
                    .onChange(of: selectedRole) { _, newValue in
                        Task { await viewModel.setRole(currentUser, role: newValue) }
                    }
                }

                Section(header: Text("Actions")) {
                    Button {
                        Task { await viewModel.resetPassword(currentUser) }
                    } label: {
                        Label("Send password reset", systemImage: "envelope.badge")
                    }
                    .disabled(viewModel.isPerformingAction)

                    if currentUser.status == .disabled {
                        Button {
                            Task { await viewModel.activate(currentUser) }
                        } label: {
                            Label("Activate user", systemImage: "person.crop.circle.badge.checkmark")
                        }
                    } else {
                        Button(role: .destructive) {
                            Task { await viewModel.disable(currentUser) }
                        } label: {
                            Label("Disable user", systemImage: "person.fill.xmark")
                        }
                    }
                }
            }
            .navigationTitle(currentUser.username)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { onDismiss() }
                }
            }
        }
        .onChange(of: currentUser.role) { _, newRole in
            if selectedRole != newRole {
                selectedRole = newRole
            }
        }
    }

    private var currentUser: AdminUser {
        viewModel.users.first(where: { $0.id == user.id }) ?? user
    }

    private var statusColor: Color {
        switch currentUser.status {
        case .active: return Theme.accentGreen
        case .disabled: return Theme.accentRed
        case .pending: return .orange
        }
    }
}
