import SwiftUI

struct AuthenticationFlowView: View {
    @State private var isPresentingSignup = false

    var body: some View {
        NavigationStack {
            LoginView(onSignup: {
                isPresentingSignup = true
            })
            .navigationDestination(isPresented: $isPresentingSignup) {
                SignupView()
            }
            .toolbar(.hidden, for: .automatic)
        }
    }
}

#Preview {
    AuthenticationFlowView()
        .environmentObject(SessionStore(previewState: .loggedOut))
}
