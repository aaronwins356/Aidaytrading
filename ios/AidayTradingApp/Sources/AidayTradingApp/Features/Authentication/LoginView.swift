import SwiftUI

struct LoginView: View {
    @EnvironmentObject private var session: SessionStore
    @State private var username: String = ""
    @State private var password: String = ""
    @State private var isSubmitting = false
    @State private var isPresentingResetSheet = false
    @State private var resetEmail: String = ""

    let onSignup: () -> Void

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Welcome back")
                        .font(.largeTitle.bold())
                        .foregroundStyle(.white)
                    Text("Sign in to monitor live equity, trades, and risk controls.")
                        .font(.callout)
                        .foregroundStyle(.white.opacity(0.7))
                }
                VStack(spacing: 16) {
                    inputField(title: "Username", text: $username, icon: "person", isSecure: false)
                    inputField(title: "Password", text: $password, icon: "lock", isSecure: true)
                }
                .padding()
                .background(Theme.cardBackground, in: RoundedRectangle(cornerRadius: 16))
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(Theme.accentGreen.opacity(0.4), lineWidth: 1)
                )

                Button {
                    submit()
                } label: {
                    if isSubmitting {
                        ProgressView()
                            .tint(.white)
                    } else {
                        Text("Secure login")
                            .bold()
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(Theme.accentGreen)
                .disabled(!formIsValid || isSubmitting)

                HStack {
                    Button("Forgot password?") {
                        resetEmail = ""
                        isPresentingResetSheet = true
                    }
                    .tint(.white)
                    Spacer()
                    Button("Request access") {
                        onSignup()
                    }
                    .tint(Theme.accentGreen)
                }
                .font(.footnote)
            }
            .padding(.horizontal, 24)
            .padding(.top, 80)
            .padding(.bottom, 32)
        }
        .background(Theme.background.ignoresSafeArea())
        .customNavigationBar(title: "AidayTrading")
        .sheet(isPresented: $isPresentingResetSheet) {
            resetSheet
        }
    }

    private var formIsValid: Bool {
        !username.isEmpty && !password.isEmpty
    }

    private func inputField(title: String, text: Binding<String>, icon: String, isSecure: Bool) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.footnote)
                .foregroundStyle(.white.opacity(0.7))
            HStack {
                Image(systemName: icon)
                    .foregroundStyle(Theme.accentGreen)
                if isSecure {
                    SecureField(title, text: text)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .foregroundStyle(.white)
                } else {
                    TextField(title, text: text)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .foregroundStyle(.white)
                }
            }
            .padding()
            .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
        }
    }

    private func submit() {
        guard formIsValid else { return }
        isSubmitting = true
        session.registerInteraction()
        Task {
            await session.login(username: username, password: password)
            await MainActor.run {
                isSubmitting = false
            }
        }
    }

    @ViewBuilder
    private var resetSheet: some View {
        NavigationStack {
            Form {
                Section("Reset password") {
                    TextField("Account email", text: $resetEmail)
                        .keyboardType(.emailAddress)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }
                Section {
                    Button("Send reset link") {
                        session.requestPasswordReset(email: resetEmail)
                        isPresentingResetSheet = false
                    }
                    .disabled(resetEmail.isEmpty || !resetEmail.contains("@"))
                }
            }
            .navigationTitle("Forgot password")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") {
                        isPresentingResetSheet = false
                    }
                }
            }
        }
    }
}

#Preview {
    LoginView(onSignup: {})
        .environmentObject(SessionStore(previewState: .loggedOut))
}
