
from flask import Flask, render_template, request
from recommendation import load_data, filter_services, get_cluster_plot_data
import plotly.express as px

app = Flask(__name__)
df, providers, types = load_data()

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    cluster_plot_html = ""
    selected_provider = ""
    selected_type = ""
    min_cpu = min_memory = min_bandwidth = 0
    use_dbscan = use_lda = optimize = False
    eps = 0.5
    min_samples = 3
    show_count = 10

    if request.method == "POST":
        if request.form.get("action") == "load_more":
            show_count = int(request.form.get("show_count", 10)) + 10
        else:
            show_count = 10
            selected_provider = request.form.get("provider", "")
            selected_type = request.form.get("type", "")
            min_cpu = float(request.form.get("min_cpu", 0))
            min_memory = float(request.form.get("min_memory", 0))
            min_bandwidth = float(request.form.get("min_bandwidth", 0))
            use_dbscan = request.form.get("dbscan") == "on"
            use_lda = request.form.get("lda") == "on"
            optimize = request.form.get("optimize") == "on"
            eps = float(request.form.get("eps", 0.5))
            min_samples = int(request.form.get("min_samples", 3))

            results = filter_services(
                df,
                provider=selected_provider,
                service_type=selected_type,
                min_cpu=min_cpu,
                min_memory=min_memory,
                min_bandwidth=min_bandwidth,
                use_dbscan=use_dbscan,
                use_lda=use_lda,
                optimize=optimize,
                eps=eps,
                min_samples=min_samples
            )

            if use_dbscan:
                plot_df = get_cluster_plot_data(df, eps=eps, min_samples=min_samples)
                fig = px.scatter(
                    plot_df,
                    x="PCA1",
                    y="PCA2",
                    color=plot_df["Cluster"].astype(str),
                    hover_data=["Service_ID"],
                    title="DBSCAN Clusters (PCA1 vs PCA2)"
                )
                cluster_plot_html = fig.to_html(full_html=False)
        # Save the latest form state (for the UI)
        form_state = {
            "selected_provider": selected_provider,
            "selected_type": selected_type,
            "min_cpu": min_cpu,
            "min_memory": min_memory,
            "min_bandwidth": min_bandwidth,
            "use_dbscan": use_dbscan,
            "use_lda": use_lda,
            "optimize": optimize,
            "eps": eps,
            "min_samples": min_samples,
            "show_count": show_count
        }
    else:
        form_state = {
            "selected_provider": "",
            "selected_type": "",
            "min_cpu": 0,
            "min_memory": 0,
            "min_bandwidth": 0,
            "use_dbscan": False,
            "use_lda": False,
            "optimize": False,
            "eps": 0.5,
            "min_samples": 3,
            "show_count": show_count
        }

    return render_template("index.html",
        providers=providers,
        types=types,
        results=results[:form_state["show_count"]],
        show_more=len(results) > form_state["show_count"] if results else False,
        cluster_plot_html=cluster_plot_html,
        **form_state
    )

if __name__ == "__main__":
    app.run(debug=True)
