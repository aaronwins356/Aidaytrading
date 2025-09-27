import SwiftUI

struct SignupView: View {
    @EnvironmentObject private var session: SessionStore
    @State private var username: String = ""
    @State private var email: String = ""
    @State private var password: String = ""
    @State private var confirmPassword: String = ""
    @State private var isSubmitting = false

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Request secure access")
                        .font(.largeTitle.bold())
                        .foregroundStyle(.white)
                    Text("Provide contact details so we can verify your identity before enabling live trading dashboards.")
                        .font(.callout)
                        .foregroundStyle(.white.opacity(0.7))
                }

                VStack(spacing: 16) {
                    floatingField(title: "Username", text: $username, icon: "person")
                    floatingField(title: "Email", text: $email, icon: "envelope")
                        .keyboardType(.emailAddress)
                    floatingSecureField(title: "Password", text: $password, icon: "lock")
                    floatingSecureField(title: "Confirm password", text: $confirmPassword, icon: "lock.rotation")
                }
                .padding()
                .background(Theme.cardBackground, in: RoundedRectangle(cornerRadius: 16))
                .overlay(
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(Theme.accentGreen.opacity(0.4), lineWidth: 1)
                )

                VStack(alignment: .leading, spacing: 8) {
                    Text("Password requirements")
                        .font(.headline)
                        .foregroundStyle(.white)
                    ForEach(passwordStrength.messages, id: \.self) { message in
                        HStack {
                            Image(systemName: passwordStrength.isValid ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                                .foregroundStyle(passwordStrength.isValid ? Theme.accentGreen : Theme.accentRed)
                            Text(message)
                                .foregroundStyle(.white.opacity(0.8))
                                .font(.subheadline)
                        }
                    }
                }

                Button {
                    submit()
                } label: {
                    if isSubmitting {
                        ProgressView()
                            .tint(.white)
                    } else {
                        Text("Submit for approval")
                            .bold()
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(Theme.accentGreen)
                .disabled(!formIsValid || isSubmitting)
            }
            .padding(.horizontal, 24)
            .padding(.top, 80)
            .padding(.bottom, 32)
        }
        .background(Theme.background.ignoresSafeArea())
        .customNavigationBar(title: "Create account")
    }

    private var passwordStrength: PasswordStrengthValidator.ValidationResult {
        PasswordStrengthValidator.validate(password: password)
    }

    private var formIsValid: Bool {
        !username.isEmpty && email.contains("@") && passwordStrength.isValid && password == confirmPassword
    }

    private func submit() {
        guard formIsValid else { return }
        isSubmitting = true
        session.registerInteraction()
        Task {
            await session.signup(username: username, email: email, password: password)
            await MainActor.run {
                isSubmitting = false
            }
        }
    }

    private func floatingField(title: String, text: Binding<String>, icon: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.footnote)
                .foregroundStyle(.white.opacity(0.7))
            HStack {
                Image(systemName: icon)
                    .foregroundStyle(Theme.accentGreen)
                TextField(title, text: text)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .foregroundStyle(.white)
            }
            .padding()
            .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
        }
    }

    private func floatingSecureField(title: String, text: Binding<String>, icon: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.footnote)
                .foregroundStyle(.white.opacity(0.7))
            HStack {
                Image(systemName: icon)
                    .foregroundStyle(Theme.accentGreen)
                SecureField(title, text: text)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .foregroundStyle(.white)
            }
            .padding()
            .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
        }
    }
}

#Preview {
    SignupView()
        .environmentObject(SessionStore(previewState: .loggedOut))
}
