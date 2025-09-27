import SwiftUI

struct MainTabView: View {
    let context: UserSessionContext
    @EnvironmentObject private var session: SessionStore
    @EnvironmentObject private var notificationController: NotificationController

    enum Tab: Hashable {
        case home
        case calendar
        case trades
        case admin
    }

    @State private var selection: Tab = .home

    var body: some View {
        TabView(selection: $selection) {
            HomeView()
                .tabItem { Label("Home", systemImage: "chart.line.uptrend.xyaxis") }
                .tag(Tab.home)

            CalendarView()
                .tabItem { Label("Calendar", systemImage: "calendar") }
                .tag(Tab.calendar)

            TradesListView()
                .tabItem { Label("Trades", systemImage: "list.bullet.rectangle") }
                .tag(Tab.trades)

            if RoleManager.isAdmin(context.profile) {
                AdminView()
                    .tabItem { Label("Admin", systemImage: "lock.shield") }
                    .tag(Tab.admin)
            }
        }
        .sensitive()
        .tint(Theme.accentGreen)
        .onChange(of: selection) { _, _ in
            session.registerInteraction()
        }
        .onAppear {
            session.registerInteraction()
        }
        .onChange(of: notificationController.pendingTab) { _, newValue in
            guard let tab = newValue else { return }
            if RoleManager.accessibleTabs(for: context.profile).contains(tab) {
                selection = tab
            }
            _ = notificationController.consumePendingTab()
        }
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button("Logout") {
                    session.logout()
                }
            }
        }
    }
}
