//! Desktop app: a native window showing the UI in `ui/`, with the engine's
//! WebSocket server running on a background thread in the same process.
//!
//! The UI files are compiled into the binary and served over a custom
//! `seraph://` protocol, so the app needs no files next to it at runtime.

use crate::{engine, server};
use std::borrow::Cow;
use std::net::TcpListener;
use tao::dpi::LogicalSize;
use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoop};
use tao::window::WindowBuilder;
use wry::http::{header::CONTENT_TYPE, Request, Response};
use wry::{WebContext, WebViewBuilder};

macro_rules! assets {
    ($($path:literal => $mime:literal),* $(,)?) => {
        &[$(($path, $mime, include_bytes!(concat!("../ui/", $path)) as &[u8])),*]
    };
}

static ASSETS: &[(&str, &str, &[u8])] = assets! {
    "index.html" => "text/html; charset=utf-8",
    "theme.css" => "text/css; charset=utf-8",
    "app.js" => "text/javascript; charset=utf-8",
    "logo.svg" => "image/svg+xml",
    "vendor/marked.min.js" => "text/javascript; charset=utf-8",
    "vendor/purify.min.js" => "text/javascript; charset=utf-8",
    "fonts/cormorant-garamond-latin-500-normal.woff2" => "font/woff2",
    "fonts/cormorant-garamond-latin-600-normal.woff2" => "font/woff2",
    "fonts/cormorant-garamond-latin-700-normal.woff2" => "font/woff2",
    "fonts/cormorant-garamond-latin-500-italic.woff2" => "font/woff2",
    "fonts/newsreader-latin-400-normal.woff2" => "font/woff2",
    "fonts/newsreader-latin-500-normal.woff2" => "font/woff2",
    "fonts/newsreader-latin-400-italic.woff2" => "font/woff2",
    "fonts/jetbrains-mono-latin-400-normal.woff2" => "font/woff2",
    "fonts/jetbrains-mono-latin-500-normal.woff2" => "font/woff2",
};

fn serve_asset(req: Request<Vec<u8>>) -> Response<Cow<'static, [u8]>> {
    let path = req.uri().path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    match ASSETS.iter().find(|(p, _, _)| *p == path) {
        Some((_, mime, body)) => Response::builder()
            .header(CONTENT_TYPE, *mime)
            .body(Cow::Borrowed(*body))
            .unwrap(),
        None => Response::builder()
            .status(404)
            .header(CONTENT_TYPE, "text/plain")
            .body(Cow::Borrowed(&b"not found"[..]))
            .unwrap(),
    }
}

pub fn run() -> Result<(), String> {
    // WebKitGTK's DMA-BUF renderer shows a blank window on many NVIDIA setups.
    #[cfg(target_os = "linux")]
    if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }

    // Engine server on a free local port; the UI is told which one.
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| format!("can't open a local port: {e}"))?;
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    let system_prompt = engine::load_system_prompt();
    std::thread::Builder::new()
        .name("engine".into())
        .spawn(move || server::serve(listener, None, system_prompt))
        .map_err(|e| format!("can't start engine thread: {e}"))?;
    println!("Engine listening on ws://127.0.0.1:{port}");

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("SeraphByte")
        .with_inner_size(LogicalSize::new(1440.0, 960.0))
        .with_min_inner_size(LogicalSize::new(380.0, 520.0))
        .build(&event_loop)
        .map_err(|e| format!("can't open window: {e}"))?;

    // Persistent storage (conversations, settings) lives in the user's data dir.
    let data_dir = dirs::data_dir().map(|d| d.join("seraphbyte"));
    let mut context = WebContext::new(data_dir);

    #[cfg(any(target_os = "windows", target_os = "android"))]
    let start_url = "http://seraph.localhost/index.html";
    #[cfg(not(any(target_os = "windows", target_os = "android")))]
    let start_url = "seraph://localhost/index.html";

    let builder = WebViewBuilder::new_with_web_context(&mut context)
        .with_custom_protocol("seraph".into(), |_id, req| serve_asset(req))
        .with_initialization_script(format!(
            "window.__SERAPH_WS__ = 'ws://127.0.0.1:{port}';
             window.addEventListener('error', e => window.ipc.postMessage('[ui error] ' + e.message + ' (' + e.filename + ':' + e.lineno + ')'));
             window.addEventListener('unhandledrejection', e => window.ipc.postMessage('[ui error] ' + e.reason));"
        ))
        // UI errors are printed to the terminal; there is no devtools window.
        .with_ipc_handler(|req| eprintln!("{}", req.body()))
        .with_background_color((14, 10, 11, 255))
        .with_url(start_url);

    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "ios", target_os = "android")))]
    let webview = {
        use tao::platform::unix::WindowExtUnix;
        use wry::WebViewBuilderExtUnix;
        let vbox = window.default_vbox().ok_or("no GTK container for the webview")?;
        builder.build_gtk(vbox)
    };
    #[cfg(any(target_os = "windows", target_os = "macos", target_os = "ios", target_os = "android"))]
    let webview = builder.build(&window);
    let webview = webview.map_err(|e| format!("can't create webview: {e}"))?;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;
        let _keep_alive = (&window, &webview);
        if let Event::WindowEvent { event: WindowEvent::CloseRequested, .. } = event {
            // The engine thread blocks on accept(); exiting the process ends it.
            std::process::exit(0);
        }
    });
}
