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
            case .pendingApproval(let email):
                PendingApprovalView(email: email) {
                    session.refreshProfile()
                }
            case .authenticated(let context):
                MainTabView(context: context)
            }
        }
        .task {
            await session.bootstrap()
        }
        .alert(item: $session.error) { error in
            Alert(title: Text("Error"), message: Text(error.message), dismissButton: .default(Text("OK")))
        }
    }
}

#Preview {
    RootView()
        .environmentObject(SessionStore(previewState: .loggedOut))
}
