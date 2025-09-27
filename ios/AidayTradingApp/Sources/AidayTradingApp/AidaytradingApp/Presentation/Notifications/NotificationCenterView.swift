import SwiftUI

struct NotificationCenterView: View {
    @EnvironmentObject private var notificationManager: NotificationManager
    @State private var selectedFilter: Filter = .all

    enum Filter: String, CaseIterable, Identifiable {
        case all
        case reports
        case botEvents
        case system

        var id: String { rawValue }

        var title: String {
            switch self {
            case .all: return "All"
            case .reports: return "Reports"
            case .botEvents: return "Bot Events"
            case .system: return "System"
            }
        }

        func matches(_ notification: AppNotification) -> Bool {
            switch self {
            case .all:
                return true
            case .reports:
                return notification.kind == .report
            case .botEvents:
                return notification.kind == .botEvent
            case .system:
                return notification.kind == .system
            }
        }
    }

    private var filteredNotifications: [AppNotification] {
        notificationManager.notifications.filter { selectedFilter.matches($0) }
    }

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 16) {
                Picker("Filter", selection: $selectedFilter) {
                    ForEach(Filter.allCases) { filter in
                        Text(filter.title).tag(filter)
                    }
                }
                .pickerStyle(.segmented)

                if filteredNotifications.isEmpty {
                    Spacer()
                    EmptyStateView(
                        title: "No alerts",
                        message: "Notifications from pushes and live events will appear here."
                    )
                    Spacer()
                } else {
                    List {
                        ForEach(filteredNotifications) { notification in
                            NavigationLink(value: notification) {
                                NotificationRowView(notification: notification)
                            }
                            .swipeActions {
                                Button(role: .destructive) {
                                    notificationManager.delete(notificationID: notification.id)
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                            }
                        }
                    }
                    .listStyle(.insetGrouped)
                    .scrollContentBackground(.hidden)
                    .background(Theme.background)
                }
            }
            .padding()
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            .background(Theme.background.ignoresSafeArea())
            .navigationTitle("Notifications")
            .navigationDestination(for: AppNotification.self) { notification in
                NotificationDetailView(notification: notification)
            }
        }
        .preferredColorScheme(.dark)
    }
}

private struct NotificationRowView: View {
    let notification: AppNotification

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Text(icon)
                .font(.title2)
                .padding(8)
                .background(Theme.cardBackground.opacity(0.6))
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))

            VStack(alignment: .leading, spacing: 6) {
                Text(notification.title)
                    .font(.headline)
                    .foregroundColor(Theme.primaryText)
                Text(notification.body)
                    .font(.subheadline)
                    .foregroundColor(Theme.secondaryText)
                Text("\(notification.centralTimestampString()) Central")
                    .font(.caption)
                    .foregroundColor(Theme.secondaryText.opacity(0.8))
            }
            Spacer()
        }
        .padding(.vertical, 8)
        .accessibilityElement(children: .combine)
    }

    private var icon: String {
        if notification.kind == .botEvent {
            if let running = notification.payloadDictionary?["running"] as? Bool {
                return running ? "🚀" : "✋"
            }
            return "🚀"
        }
        if notification.kind == .report {
            return "📊"
        }
        return "🛰️"
    }
}

private struct NotificationDetailView: View {
    let notification: AppNotification

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text(notification.title)
                    .font(.title2.bold())
                    .foregroundColor(Theme.primaryText)
                Text(notification.body)
                    .font(.body)
                    .foregroundColor(Theme.secondaryText)
                Text("Received \(notification.centralTimestampString()) Central")
                    .font(.footnote)
                    .foregroundColor(Theme.secondaryText)
                Divider()
                Text("Payload")
                    .font(.headline)
                    .foregroundColor(Theme.primaryText)
                Text(payloadString)
                    .font(.system(.body, design: .monospaced))
                    .foregroundColor(Theme.secondaryText)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .background(Theme.cardBackground)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            }
            .padding()
        }
        .background(Theme.background.ignoresSafeArea())
        .navigationTitle("Alert Detail")
        .navigationBarTitleDisplayMode(.inline)
    }

    private var payloadString: String {
        guard let dictionary = notification.payloadDictionary,
              let data = try? JSONSerialization.data(withJSONObject: dictionary, options: [.prettyPrinted]),
              let string = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return string
    }
}
