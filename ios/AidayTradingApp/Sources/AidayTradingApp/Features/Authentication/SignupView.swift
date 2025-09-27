import SwiftUI

struct SignupView: View {
    @EnvironmentObject private var session: SessionStore
    @State private var username: String = ""
    @State private var email: String = ""
    @State private var password: String = ""
    @State private var confirmPassword: String = ""
    @State private var isSubmitting = false

    var body: some View {
        Form {
            Section(header: Text("Account details")) {
                TextField("Username", text: $username)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                TextField("Email", text: $email)
                    .keyboardType(.emailAddress)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                SecureField("Password", text: $password)
                SecureField("Confirm Password", text: $confirmPassword)
            }

            Section(footer: passwordRequirementsView) {
                Button(action: signup) {
                    if isSubmitting {
                        ProgressView()
                    } else {
                        Text("Request access")
                    }
                }
                .disabled(!formIsValid || isSubmitting)
            }
        }
        .navigationTitle("Create account")
    }

    private var formIsValid: Bool {
        !username.isEmpty && email.contains("@") && passwordStrength.isValid && password == confirmPassword
    }

    private var passwordStrength: PasswordStrengthValidator.ValidationResult {
        PasswordStrengthValidator.validate(password: password)
    }

    @ViewBuilder
    private var passwordRequirementsView: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Password must contain:")
                .font(.footnote)
            ForEach(passwordStrength.messages, id: \.self) { message in
                Label(message, systemImage: passwordStrength.isValid ? "checkmark.circle" : "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(passwordStrength.isValid ? .green : .orange)
            }
        }
    }

    private func signup() {
        guard formIsValid else { return }
        isSubmitting = true
        Task {
            await session.signup(username: username, email: email, password: password)
            await MainActor.run {
                isSubmitting = false
            }
        }
    }
}

#Preview {
    SignupView()
        .environmentObject(SessionStore(previewState: .loggedOut))
}
