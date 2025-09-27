import SwiftUI

struct HomeView: View {
    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 16) {
                Text("Dashboards")
                    .font(.title2)
                    .bold()
                Text("Connects to backend metrics once available.")
                    .font(.body)
                Spacer()
            }
            .padding()
            .navigationTitle("Overview")
        }
    }
}
