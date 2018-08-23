package main

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/gorilla/mux"
)

func makeMuxRouter() http.Handler {
	muxRouter := mux.NewRouter()
	muxRouter.HandleFunc("/definition/{id}", handlerGetDefinitionID).Methods("GET")
	muxRouter.HandleFunc("/definition", handlerPostDefinition).Methods("POST")
	muxRouter.HandleFunc("/definition", handlerGetDefinition).Methods("GET")
	muxRouter.HandleFunc("/definition", handlerPutDefinition).Methods("PUT")
	return muxRouter
}

func handlerGetDefinition(w http.ResponseWriter, r *http.Request) {
	// Retrieve people from postgresql database using our `store` interface variable's
	// `func (*dbstore) GetPerson` pointer receiver method defined in `store.go` file
	// personList, err := store.GetPerson()

	// // Convert the `personList` variable to JSON
	// personListBytes, err := json.Marshal(personList)
	// if err != nil {
	// 	fmt.Println(fmt.Errorf("Error: %v", err))
	// 	w.WriteHeader(http.StatusInternalServerError)
	// 	return
	// }

	// // Write JSON list of persons to response
	// w.Write(personListBytes)

	fmt.Fprintln(w, "Not implemented yet get def")
}

func handlerPostDefinition(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "Not implemented yet postdef")
}
func handlerGetDefinitionID(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "Not implemented yet getdefID")
}
func handlerPutDefinition(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "Not implemented yet putdef")
}

func respondWithError(w http.ResponseWriter, r *http.Request, code int, msg string) {
	respondWithJSON(w, code, map[string]string{"error": msg})
}

func respondWithJSON(w http.ResponseWriter, code int, payload interface{}) {
	response, err := json.MarshalIndent(payload, "", " ")
	if err != nil {
		http.Error(w, "HTTP 500: Internal Server Error", http.StatusInternalServerError)
		// w.WriteHeader(http.StatusInternalServerError)
		// w.Write([]byte("HTTP 500: Internal Server Error"))
		return
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	w.Write(response)
}
