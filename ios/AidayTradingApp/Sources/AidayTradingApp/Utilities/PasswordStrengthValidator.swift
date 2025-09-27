import Foundation

enum PasswordStrengthValidator {
    struct ValidationResult {
        let isValid: Bool
        let messages: [String]
    }

    static func validate(password: String) -> ValidationResult {
        let requirements: [(String, (String) -> Bool)] = [
            ("At least 8 characters", { $0.count >= 8 }),
            ("One uppercase letter", { $0.range(of: "[A-Z]", options: .regularExpression) != nil }),
            ("One lowercase letter", { $0.range(of: "[a-z]", options: .regularExpression) != nil }),
            ("One number", { $0.range(of: "[0-9]", options: .regularExpression) != nil })
        ]

        let failures = requirements.compactMap { requirement -> String? in
            requirement.1(password) ? nil : requirement.0
        }

        let messages = failures.isEmpty ? ["Strong password"] : failures
        return ValidationResult(isValid: failures.isEmpty, messages: messages)
    }
}
