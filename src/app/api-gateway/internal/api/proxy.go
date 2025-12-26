package api

import (
	"encoding/json"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"time"
)

func newReverseProxy(targetRaw string, stripPrefix string) *httputil.ReverseProxy {
	target := mustParseURL(targetRaw)
	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.Director = func(req *http.Request) {
		originalHost := req.Host
		req.URL.Scheme = target.Scheme
		req.URL.Host = target.Host

		path := req.URL.Path
		if stripPrefix != "" {
			path = strings.TrimPrefix(path, stripPrefix)
			if path == "" {
				path = "/"
			}
		}
		req.URL.Path = singleJoiningSlash(target.Path, path)

		req.Host = target.Host
		req.Header.Set("X-Forwarded-Host", originalHost)
		req.Header.Set("X-Forwarded-Proto", forwardedProto(req))

		if clientIP, _, err := net.SplitHostPort(req.RemoteAddr); err == nil {
			prior := req.Header.Get("X-Forwarded-For")
			if prior == "" {
				req.Header.Set("X-Forwarded-For", clientIP)
			} else {
				req.Header.Set("X-Forwarded-For", prior+", "+clientIP)
			}
		}
	}
	proxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
		log.Printf("proxy error %s %s: %v", r.Method, r.URL.Path, err)
		writeJSON(w, http.StatusBadGateway, map[string]string{
			"error": "upstream unavailable",
		})
	}
	proxy.FlushInterval = 50 * time.Millisecond
	return proxy
}

func singleJoiningSlash(a, b string) string {
	slashA := strings.HasSuffix(a, "/")
	slashB := strings.HasPrefix(b, "/")
	if slashA && slashB {
		return a + b[1:]
	}
	if !slashA && !slashB {
		return a + "/" + b
	}
	return a + b
}

func forwardedProto(req *http.Request) string {
	if req.TLS != nil {
		return "https"
	}
	return "http"
}

func writeJSON(w http.ResponseWriter, status int, payload any) {
	data, err := json.Marshal(payload)
	if err != nil {
		http.Error(w, "failed to encode response", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(data)
}

func mustParseURL(raw string) *url.URL {
	parsed, err := url.Parse(raw)
	if err != nil {
		log.Fatalf("invalid url %q: %v", raw, err)
	}
	if parsed.Scheme == "" || parsed.Host == "" {
		log.Fatalf("invalid url %q: missing scheme or host", raw)
	}
	return parsed
}
