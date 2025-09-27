import SwiftUI

struct MainTabView: View {
    let context: UserSessionContext
    @EnvironmentObject private var session: SessionStore
    @EnvironmentObject private var notificationManager: NotificationManager
    @EnvironmentObject private var realtimeClient: TradingWebSocketClient

    enum Tab: Hashable {
        case home
        case calendar
        case trades
        case notifications
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

            NotificationCenterView()
                .tabItem { Label("Alerts", systemImage: "bell.badge") }
                .tag(Tab.notifications)

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
            notificationManager.bind(to: realtimeClient)
        }
        .onChange(of: notificationManager.pendingTab) { _, newValue in
            guard let tab = newValue else { return }
            if RoleManager.accessibleTabs(for: context.profile).contains(tab) {
                selection = tab
            }
            _ = notificationManager.consumePendingTab()
        }
        .toolbar {
            ToolbarItem(placement: .navigationBarTrailing) {
                Button("Logout") {
                    session.logout()
                }
            }
        }
        .overlay(alignment: .top) {
            if let banner = notificationManager.activeBanner {
                NotificationBannerView(notification: banner)
                    .transition(.move(edge: .top).combined(with: .opacity))
                    .padding(.top, 8)
            }
        }
        .animation(.spring(response: 0.5, dampingFraction: 0.8), value: notificationManager.activeBanner)
    }
}
