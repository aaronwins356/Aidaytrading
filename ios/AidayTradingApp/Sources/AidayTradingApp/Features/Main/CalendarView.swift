import SwiftUI

struct CalendarView: View {
    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                Image(systemName: "calendar")
                    .font(.system(size: 48))
                    .foregroundStyle(.accent)
                Text("Calendar integrations coming soon.")
                    .font(.body)
            }
            .padding()
            .navigationTitle("Calendar")
        }
    }
}
