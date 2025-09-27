import Foundation

extension Decimal {
    func currencyString(currencyCode: String = "USD", locale: Locale = Locale.current) -> String {
        let formatter = NumberFormatter()
        formatter.locale = locale
        formatter.numberStyle = .currency
        formatter.currencyCode = currencyCode
        formatter.maximumFractionDigits = 2
        return formatter.string(from: self as NSDecimalNumber) ?? "—"
    }

    func signedPercentString(locale: Locale = Locale.current, fractionDigits: Int = 2) -> String {
        let formatter = NumberFormatter()
        formatter.locale = locale
        formatter.numberStyle = .percent
        formatter.maximumFractionDigits = fractionDigits
        formatter.minimumFractionDigits = fractionDigits
        formatter.positivePrefix = "+"
        formatter.negativePrefix = "−"
        return formatter.string(from: (self as NSDecimalNumber).dividing(by: 100)) ?? "—"
    }
}
