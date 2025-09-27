import LocalAuthentication

protocol BiometricAuthenticating {
    func authenticate(reason: String) async throws
}

struct BiometricAuthenticator: BiometricAuthenticating {
    func authenticate(reason: String) async throws {
        let context = LAContext()
        context.localizedFallbackTitle = "Use Passcode"
        var error: NSError?
        guard context.canEvaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, error: &error) else {
            throw error ?? LAError(.biometryNotAvailable)
        }
        return try await withCheckedThrowingContinuation { continuation in
            context.evaluatePolicy(.deviceOwnerAuthenticationWithBiometrics, localizedReason: reason) { success, authError in
                if success {
                    continuation.resume()
                } else {
                    let failure = authError ?? LAError(.authenticationFailed)
                    continuation.resume(throwing: failure)
                }
            }
        }
    }
}
