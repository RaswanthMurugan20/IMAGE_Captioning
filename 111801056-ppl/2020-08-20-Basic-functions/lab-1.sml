(*Q.1*)
fun curry f a b c = f(a,b,c)
		     
fun uncurry f(a,b,c) = f a b c

(*Q.2*)			 
fun fst(x,y) = x
		   
fun snd(x,y) = y

(*Q.3*)
fun length [] = 0
  | length (x::xs) = 1 + length xs
				
(*Q.4*)
fun reverse [] = []
  | reverse (x::xs) =
    let
        fun append [] x = [x]
 	  | append (x::xs) y = x :: (append xs y)
    in append (reverse xs) x
    end;

(*Q.5*)			     
fun fib n =
    let
	fun fibrec a b 0 = a
	  | fibrec a b n = fibrec b (a+b) (n-1)
    in fibrec 1 1 n
    end;				
