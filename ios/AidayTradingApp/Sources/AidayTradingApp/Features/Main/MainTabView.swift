import SwiftUI

struct MainTabView: View {
    let context: UserSessionContext
    @EnvironmentObject private var session: SessionStore

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
                .tabItem {
                    Label("Home", systemImage: "chart.line.uptrend.xyaxis")
                }
                .tag(Tab.home)

            CalendarView()
                .tabItem {
                    Label("Calendar", systemImage: "calendar")
                }
                .tag(Tab.calendar)

            TradesView()
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
        .onChange(of: selection) { _, _ in
            session.registerInteraction()
        }
        .onAppear {
            session.registerInteraction()
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
}
