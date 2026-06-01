from shiny import App, ui

app_ui = ui.page_fluid(

    ui.tags.head(
        ui.tags.style("""
            body {
                margin: 0;
                font-family: Arial, sans-serif;
                background-color: #081028;
                color: white;
            }

            .sidebar {
                width: 260px;
                height: 100vh;
                position: fixed;
                background-color: #020B23;
                padding: 20px;
            }

            .logo {
                font-size: 32px;
                font-weight: bold;
                margin-bottom: 40px;
                color: white;
            }

            .nav-item {
                padding: 15px;
                border-radius: 12px;
                margin-bottom: 10px;
                background-color: #1B2740;
                cursor: pointer;
                font-size: 18px;
            }

            .main {
                margin-left: 300px;
                padding: 40px;
            }

            .title {
                font-size: 52px;
                font-weight: bold;
            }

            .subtitle {
                color: #9CA3AF;
                margin-bottom: 30px;
                font-size: 20px;
            }

            .card {
                background-color: #1B2740;
                padding: 30px;
                border-radius: 20px;
                width: 700px;
            }

            .input-box {
                width: 100%;
                padding: 15px;
                border-radius: 12px;
                border: none;
                margin-top: 10px;
                margin-bottom: 20px;
                font-size: 16px;
            }

            .btn {
                background-color: #1DA1F2;
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 12px;
                font-size: 18px;
                cursor: pointer;
            }

            .recent {
                position: absolute;
                right: 80px;
                top: 120px;
                width: 300px;
            }

            .recent-card {
                background-color: #1B2740;
                border-radius: 18px;
                padding: 15px;
                margin-top: 20px;
            }
        """)
    ),

    ui.div(
        {"class": "sidebar"},

        ui.div("VidDigest", class_="logo"),

        ui.div("📊 Summarizer", class_="nav-item"),
        ui.div("🕘 History", class_="nav-item"),
        ui.div("📈 Stats", class_="nav-item"),
    ),

    ui.div(
        {"class": "main"},

        ui.div("Video Summarizer", class_="title"),

        ui.div(
            "Extract key insights from any YouTube video in seconds.",
            class_="subtitle"
        ),

        ui.div(
            {"class": "card"},

            ui.h3("YouTube URL"),

            ui.input_text(
                "youtube_url",
                "",
                placeholder="https://youtube.com/watch?v=..."
            ),

            ui.br(),

            ui.h3("Summary Detail"),

            ui.input_select(
                "summary_type",
                "",
                {
                    "Bullet Points": "Bullet Points",
                    "Detailed Analysis": "Detailed Analysis",
                    "Beginner Notes": "Beginner Notes",
                    "Interview Notes": "Interview Notes"
                }
            ),

            ui.br(),

            ui.input_action_button(
                "summarize",
                "✨ Summarize"
            )
        ),

        ui.div(
            {"class": "recent"},

            ui.h2("Recent Summaries"),

            ui.div(
                {"class": "recent-card"},

                ui.h4("YouTube Video"),
                ui.p("Bullet Points")
            )
        )
    )
)

def server(input, output, session):
    pass

app = App(app_ui, server)