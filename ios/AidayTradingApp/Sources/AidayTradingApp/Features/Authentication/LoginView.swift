import SwiftUI

struct LoginView: View {
    @EnvironmentObject private var session: SessionStore
    @State private var email: String = ""
    @State private var password: String = ""
    @State private var isSubmitting = false

    let onSignup: () -> Void

    var body: some View {
        Form {
            Section(header: Text("Credentials")) {
                TextField("Email", text: $email)
                    .keyboardType(.emailAddress)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                SecureField("Password", text: $password)
            }

            Section {
                Button(action: login) {
                    if isSubmitting {
                        ProgressView()
                    } else {
                        Text("Login")
                    }
                }
                .disabled(!formIsValid || isSubmitting)

                Button("Create account") {
                    onSignup()
                }
                .buttonStyle(.borderless)
            }
        }
        .navigationTitle("Welcome back")
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("Reset form") {
                    email = ""
                    password = ""
                }
            }
        }
    }

    private var formIsValid: Bool {
        !email.isEmpty && !password.isEmpty
    }

    private func login() {
        guard formIsValid else { return }
        isSubmitting = true
        Task {
            await session.login(email: email, password: password)
            await MainActor.run {
                isSubmitting = false
            }
        }
    }
}

#Preview {
    LoginView(onSignup: {})
        .environmentObject(SessionStore(previewState: .loggedOut))
}
