import SwiftUI

struct AdminView: View {
    @EnvironmentObject private var notificationManager: NotificationManager

    var body: some View {
        NavigationStack {
            List {
                notificationsSection
                Section("Risk configuration") {
                    Label("Manage risk rules", systemImage: "shield.lefthalf.filled")
                }
                Section("User management") {
                    Label("Approve or disable users", systemImage: "person.crop.circle.badge.checkmark")
                }
            }
            .navigationTitle("Admin")
        }
    }

    private var notificationsSection: some View {
        Section("Notifications") {
            Toggle(isOn: Binding(
                get: { notificationManager.preferences.botEventsEnabled },
                set: { notificationManager.setBotEvents(enabled: $0) }
            )) {
                Label("Bot events", systemImage: "bolt.circle")
            }
            .disabled(!notificationManager.canModifyPreferences)

            Toggle(isOn: Binding(
                get: { notificationManager.preferences.reportsEnabled },
                set: { notificationManager.setReports(enabled: $0) }
            )) {
                Label("Scheduled reports", systemImage: "clock.badge.checkmark")
            }
            .disabled(!notificationManager.canModifyPreferences)

            Button {
                Task { await notificationManager.unregisterDevice() }
            } label: {
                Label("Unregister this device", systemImage: "bell.slash")
            }
            .disabled(!notificationManager.canModifyPreferences)
        }
    }
}
