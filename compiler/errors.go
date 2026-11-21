package compiler

import (
	"fmt"
	"go/token"
	"strings"
)

// Error is a compile error at a source position.
type Error struct {
	Pos token.Position
	Msg string
	at  token.Pos // where, in the file set (generics.go tells the copies apart by it)
}

func (e *Error) Error() string {
	if e.Pos.IsValid() {
		return fmt.Sprintf("%s: %s", e.Pos, e.Msg)
	}
	return e.Msg
}

// ErrorList is several compile errors, reported together.
type ErrorList []*Error

func (l ErrorList) Error() string {
	var b strings.Builder
	for i, e := range l {
		if i > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(e.Error())
	}
	return b.String()
}
