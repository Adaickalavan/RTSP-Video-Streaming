package main

import (
	"fmt"
)

func main() {

	fmt.Println("yess")
}

// func run() error {
// 	mux := makeMuxRouter()
// 	httpAddr := os.Getenv("ADDR")
// 	log.Println("Listening on ", httpAddr)
// 	s := &http.Server{
// 		Addr:           ":" + httpAddr,
// 		Handler:        mux,
// 		ReadTimeout:    10 * time.Second,
// 		WriteTimeout:   10 * time.Second,
// 		MaxHeaderBytes: 1 << 20,
// 	}

// 	if err := s.ListenAndServe(); err != nil {
// 		return err
// 	}
// 	return nil
// }
