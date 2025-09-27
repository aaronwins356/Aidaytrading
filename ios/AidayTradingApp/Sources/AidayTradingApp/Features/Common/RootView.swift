import SwiftUI

struct RootView: View {
    @EnvironmentObject private var session: SessionStore

    var body: some View {
        Group {
            switch session.state {
            case .loading:
                ProgressView("Loading session…")
            case .loggedOut:
                AuthenticationFlowView()
            case .pendingApproval(let context):
                PendingApprovalView(context: context) {
                    session.refreshProfile()
                }
            case .authenticated(let context):
                MainTabView(context: context)
            }
        }
        .background(Theme.background.ignoresSafeArea())
        .task {
            await session.bootstrap()
        }
        .alert(item: $session.alert) { alert in
            Alert(title: Text(alert.title), message: Text(alert.message), dismissButton: .default(Text("OK")))
        }
    }
}

#Preview {
    RootView()
        .environmentObject(SessionStore(previewState: .loggedOut))
        .environmentObject(NotificationController())
}
