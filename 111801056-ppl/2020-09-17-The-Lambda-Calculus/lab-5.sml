(*

1.
Define a type expr to capture this abstract syntax using ML data types with variables represented as strings type var = string datatype expr = ...

*)

datatype expr = NIL
	| var of string
	| compose of expr*expr
	| lambda of string*expr
			     
(*

2.
Write a function fresh : string list -> string which will produce a fresh variable name, i.e. given xs : string list, the strin fresh xs will be different from all the strings in xs. 
Hint: Use Cantors diagonalisation

*)
			     
fun fresh xs =
    let fun big_string nil = ""
	| big_string (x::xs) = 
		let val a = big_string xs
		in
			if size(x) > size(a) then x else a
		end
		    
    in  concat ((big_string xs) :: ["Z"])
    end


(*

3.
Consider an expression like fn x => x y. Here x is a bound variable because it occurs in the shadow of the binder fn x => ... However, y is a free variable. Write a functions free : expr -> var list to compute the list of free and all the Hint: First write down a formal definition for what is the set of free variable

*)

(*Set Operations*)
fun member((x:string),[]) = false 
  | member((x:string),b::y) = if (x = b) then true
		     else member(x,y)
(*Set difference*)
fun setdif ([],xs) = []
  | setdif (xs,[]) = xs 
  | setdif (x::xs, ys) = if member(x,ys) then setdif(xs,ys)
			 else x::setdif(xs,ys)
(*Set Union*)
fun union ([],xs) = xs
  | union (xs,[]) = xs 
  | union (x::xs, ys) = if member(x,ys) then union(xs,ys)
			 else x::union(xs,ys)
fun freelst ls (lambda(x, xs)) = setdif((freelst ls xs),[x])
	| freelst ls (compose(x, y)) = union((freelst ls x),(freelst ls y))
	| freelst ls (var(x)) = if (member(x,ls) = true) then [] else [x]
	| freelst ls NIL = ls 


val free = freelst []
		   
(* example *)
		   
val e1  = lambda("x",  lambda("t",compose(var("x"),var("t"))))
val freeway = free e1
		   
(*

4.
Write a function subst : var * expr -> expr -> expr where subst (x,N) M substitutes all free occurrence of x in M with N. In mathematical notation it is written as M [x:=N].

*)
		 
fun subst (y,N) (var(x)) = if (x = y) then N else (var(x))
  | subst (y,N) (lambda(a, b)) = if (y = a) then (lambda(a, b)) else (lambda(a, (subst (y,N) b)))
  | subst (y,N) (compose(a,b)) = compose((subst (y,N) a), (subst (y,N) b))
  | subst ("",N) e1 = e1
  | subst (y,NIL) e1 = e1
  | subst (y,N) NIL = NIL

(* example *)
val e2 = subst ("y",(var("t"))) e1
val free2 = free e2
		 
fun varls ls (var(x)) = if (member(x,ls) = true) then [] else [x]
  | varls ls (lambda(x,e)) = union([x],(varls ls e))
  | varls ls (compose(e1,e2)) =  union((union(ls,(varls ls e1))),(varls ls e2))
  | varls ls NIL = ls

fun vlist M N = union((varls [] M),(varls [] N))
			
fun bdlist (e1:expr) = setdif((varls [] e1), (freelst [] e1))

fun fresh_ tlst n = if n = 1 then [(fresh tlst)]
		    else if n<= 0 then []
		    else (fresh tlst)::(fresh_ ((fresh tlst)::tlst) (n-1)) 

fun len (x::xs) = 1 + (len xs)
  | len [] = 0 

val qwer = fresh_ (vlist (lambda("t",compose(var("x"),var("t")))) (var("t"))) (len (bdlist (lambda("t",compose(var("x"),var("t"))))))

fun enc vl (x::xs) = if (vl = x) then 0  else (1 + (enc vl xs))
    
fun dnc idx (y::ys) = if (idx = 0) then y else (dnc (idx-1) ys)

				   
			 
fun freshen (qwer,bs) (var(x)) = if (member(x,bs) = true)
			         then var((dnc (enc x bs) qwer))
				 else var(x)
					 
  | freshen (qwer,bs) (lambda(x,e1)) = if (member(x,bs) = true)
				       then lambda((dnc (enc x bs) qwer), freshen (qwer, bs) e1)
				       else lambda(x, freshen (qwer, bs) e1)
						  
  | freshen (qwer,bs) (compose(e1,e2)) = compose((freshen (qwer, bs) e1),(freshen (qwer,bs) e1))
  | freshen (_,[]) _ = NIL
  | freshen (_,_) NIL = NIL
	 
		
(* substitution *)
			    
val qsc = subst ("x",var("t")) (freshen (qwer,(bdlist (lambda("t",compose(var("x"),var("t")))))) (lambda("t",compose(var("x"),var("t")))))

	       
	       

	 



	
		       


				
			       
	 
		 
