import SwiftUI

struct MainTabView: View {
    let context: UserSessionContext
    private let reportingService: ReportingServiceProtocol
    @EnvironmentObject private var session: SessionStore
    @EnvironmentObject private var notificationController: NotificationController

    enum Tab: Hashable {
        case home
        case calendar
        case trades
        case admin
    }

    @State private var selection: Tab = .home

    init(context: UserSessionContext, reportingService: ReportingServiceProtocol = ReportingService()) {
        self.context = context
        self.reportingService = reportingService
    }

    var body: some View {
        TabView(selection: $selection) {
            HomeView(context: context, reportingService: reportingService)
                .tabItem {
                    Label("Home", systemImage: "chart.line.uptrend.xyaxis")
                }
                .tag(Tab.home)

            CalendarView(context: context, reportingService: reportingService)
                .tabItem {
                    Label("Calendar", systemImage: "calendar")
                }
                .tag(Tab.calendar)

            TradesView(context: context, reportingService: reportingService)
                .tabItem {
                    Label("Trades", systemImage: "list.bullet.rectangle")
                }
                .tag(Tab.trades)

            if context.profile.role == .admin {
                AdminView()
                    .tabItem {
                        Label("Admin", systemImage: "lock.shield")
                    }
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
            selection = tab
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

#Preview {
    let profile = UserProfile(
        id: UUID(),
        username: "Admin",
        email: "admin@example.com",
        role: .admin,
        approvalStatus: .approved
    )
    let tokens = AuthTokens(accessToken: "token", refreshToken: "refresh", accessTokenExpiry: .distantFuture)
    return MainTabView(context: UserSessionContext(profile: profile, tokens: tokens))
        .environmentObject(SessionStore(previewState: .authenticated(UserSessionContext(profile: profile, tokens: tokens))))
        .environmentObject(NotificationController())
}
