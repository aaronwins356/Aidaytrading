import SwiftUI

struct PendingApprovalView: View {
    let context: SessionStore.PendingApprovalContext
    let onRefresh: () -> Void
    @EnvironmentObject private var session: SessionStore
    @State private var isRefreshing = false

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                Image(systemName: "hourglass.circle.fill")
                    .font(.system(size: 72))
                    .foregroundStyle(Theme.accentGreen)
                    .padding(.top, 32)
                VStack(spacing: 8) {
                    Text("Your account is awaiting review")
                        .font(.title2)
                        .bold()
                        .foregroundStyle(.white)
                        .multilineTextAlignment(.center)
                    Text("Thanks for joining, \(context.username).").foregroundStyle(.white.opacity(0.8))
                    Text("We notify our trading desk immediately. You'll receive an email at \(context.email) once we're done.")
                        .font(.callout)
                        .foregroundStyle(.white.opacity(0.7))
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
                VStack(alignment: .leading, spacing: 12) {
                    Label("Status: Pending manual approval", systemImage: "clock")
                        .foregroundStyle(.white)
                    Label("Security: Trading data stays locked until approval", systemImage: "lock.shield")
                        .foregroundStyle(.white)
                    Label("Checks run automatically every 30 seconds", systemImage: "arrow.clockwise")
                        .foregroundStyle(.white)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .background(Theme.cardBackground, in: RoundedRectangle(cornerRadius: 16))
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(Theme.accentGreen.opacity(0.4), lineWidth: 1)
                )
                Button {
                    guard !isRefreshing else { return }
                    isRefreshing = true
                    session.registerInteraction()
                    Task {
                        onRefresh()
                        try? await Task.sleep(nanoseconds: 500_000_000)
                        await MainActor.run {
                            isRefreshing = false
                        }
                    }
                } label: {
                    if isRefreshing {
                        ProgressView()
                            .tint(.white)
                    } else {
                        Text("Check status now")
                            .bold()
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(Theme.accentGreen)

                Button(role: .destructive) {
                    session.logout()
                } label: {
                    Text("Log out")
                        .bold()
                }
                .buttonStyle(.bordered)
                .tint(Theme.accentRed)
                .padding(.bottom, 40)
            }
            .padding(.horizontal, 24)
        }
        .background(Theme.background.ignoresSafeArea())
        .customNavigationBar(title: "Awaiting approval")
        .onAppear {
            session.registerInteraction()
        }
    }
}

#Preview {
    PendingApprovalView(
        context: SessionStore.PendingApprovalContext(username: "trader", email: "trader@example.com", tokens: nil),
        onRefresh: {}
    )
    .environmentObject(SessionStore(previewState: .pendingApproval(.init(username: "trader", email: "trader@example.com", tokens: nil))))
}
