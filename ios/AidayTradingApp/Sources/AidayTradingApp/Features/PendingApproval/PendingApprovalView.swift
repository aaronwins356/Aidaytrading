import SwiftUI

struct PendingApprovalView: View {
    let email: String
    let onRefresh: () -> Void
    @EnvironmentObject private var session: SessionStore

    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "hourglass")
                .font(.system(size: 48))
                .foregroundStyle(.accent)
            Text("Account pending approval")
                .font(.title2)
                .bold()
            Text("We have received your request. An administrator will review your account and notify you via email at \(email).")
                .font(.body)
                .multilineTextAlignment(.center)
            Button("Refresh status") {
                onRefresh()
            }
            .buttonStyle(.borderedProminent)
            Button("Logout") {
                session.logout()
            }
            .tint(.red)
        }
        .padding()
        .multilineTextAlignment(.center)
    }
}

#Preview {
    PendingApprovalView(email: "trader@example.com", onRefresh: {})
        .environmentObject(SessionStore(previewState: .pendingApproval(email: "trader@example.com")))
}
