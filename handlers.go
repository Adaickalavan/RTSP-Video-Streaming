package main

import (
	"fmt"
	"net/http"

	"github.com/gorilla/mux"
)

func makeMuxRouter() http.Handler {
	muxRouter := mux.NewRouter()
	muxRouter.HandleFunc("/definition", getDefinition).Methods("GET")
	// muxRouter.HandleFunc("/request", getWriteBlock).Methods("POST")
	return muxRouter
}

func getDefinition(w http.ResponseWriter, r *http.Request) {
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

	fmt.Println("Not implemented yet")
}
