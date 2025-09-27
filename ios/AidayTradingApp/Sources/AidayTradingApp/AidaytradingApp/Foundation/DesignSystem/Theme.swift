import SwiftUI

enum Theme {
    static let background = Color.brandBackground
    static let cardBackground = Color.brandCard
    static let accentGreen = Color.accentGreen
    static let accentRed = Color.accentRed
    static let primaryText = Color.textPrimary
    static let secondaryText = Color.textSecondary

    enum Typography {
        static let headline = Font.system(size: 24, weight: .semibold)
        static let metric = Font.system(size: 28, weight: .bold, design: .rounded)
        static let label = Font.footnote
    }
}
