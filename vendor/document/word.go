package document

import "gopkg.in/mgo.v2/bson"

//Word properties
type Word struct {
	ID      bson.ObjectId `bson:"_id" json:"id"`
	Value   string        `bson:"value" json:"value"`
	Meaning string        `bson:"meaning" json:"meaning"`
	Usage   []string      `bson:"usage" json:"usage"`
}
