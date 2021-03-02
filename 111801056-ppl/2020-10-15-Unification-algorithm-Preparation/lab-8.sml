(*Q.1.1*)
signature SIG = sig
    type symbol 
    val arity: symbol -> int
    structure Ord : ORD_KEY where type ord_key = symbol
end

(*Q.1.2*)
signature VAR = sig
    type var
    structure Ord : ORD_KEY where type ord_key = var
end

(*Q.2*)
functor Term (structure S:SIG; structure V:VAR) = struct

datatype term = v of V.var | ft of (S.symbol)*(term list) 
			 
fun occurs (v(x), (var: V.var)) = if (V.Ord.compare(x,var)) = EQUAL then true else false 
  | occurs (ft(x, (y::ys)), (var: V.var)) = if (occurs (y, var) = true) then true else
					   (occurs (ft(x,ys),var))					  
  | occurs (ft(x, []), (var: V.var)) = false 			 

(*Q.3*)

structure Telescope = RedBlackMapFn(V.Ord)
val empty = Telescope.empty
fun validterm (x: S.symbol, lst: term list): term = let exception INVALID in if (S.arity x = List.length lst) then (ft(x, lst)) else raise INVALID end

fun union ([],xs) = xs
  | union (xs,[]) = xs 
  | union (x::xs, ys) =
    let fun member ((x:V.var), []) = false 
	  | member ((x:V.var), ((b::y): V.var list)) =
	    if (V.Ord.compare(x,b) = EQUAL) then true
	    else member(x,y) 
    in
	if member(x,ys) then union(xs,ys)
	else x::union(xs,ys)

    end


(*the function to find all vaeriables related to th e givene term directly and indirectly*)
fun varlist (ls: V.var list) (tele: term Telescope.map) (SOME(ft(f,(t::ts)))) = union((varlist ls tele (SOME(t))), (varlist ls tele (SOME((ft (f, ts))))))					 
      | varlist ls tele (SOME (ft(_,[]))) = ls
      | varlist ls tele (SOME (v(x))) = varlist (union([x],ls)) tele (Telescope.find(tele,x))
      | varlist ls tele NONE = ls

fun addterm (tele: term Telescope.map) (x: V.var) (t: term) =
    let fun checker (k: V.var) ((y::ys): V.var list) =
	    if (V.Ord.compare(k,y) = EQUAL) then false
	    else checker k ys
	  | checker k [] = true  
    in 
	(case t of
	    v(y) => if (V.Ord.compare(x,y) = EQUAL) then SOME tele
		      else ( if (checker x (varlist [] tele (SOME(t)))) = true
			     then (SOME (Telescope.insert(tele,x,t)))
			     else NONE  )

	  | ft(f,ts) => if (checker x (varlist [] tele (SOME(t)))) = true
			     then (SOME (Telescope.insert(tele,x,t)))
			     else NONE		
        )									      								       
    end

end

(* Example *)
				
datatype symbols = fz | fk | fx | fy | fh
datatype variables = x1 | x2 | x3 | x4 | x5
			      
structure S: SIG = struct
type symbol = symbols
fun arity fz = 0
  | arity fk = 0 
  | arity fh = 1
  | arity fx = 2
  | arity fy = 2
		   
structure Ord: ORD_KEY = struct
type ord_key = symbol
fun compare (fz, fz) = EQUAL
  | compare (fz, fk) = LESS
  | compare (fz, fx) = LESS
  | compare (fz, fy) = LESS
  | compare (fz, fh) = LESS
  | compare (fk, fz) = GREATER
  | compare (fk, fk) = EQUAL
  | compare (fk, fx) = LESS
  | compare (fk, fy) = LESS
  | compare (fk, fh) = LESS
  | compare (fx, fz) = GREATER
  | compare (fx, fk) = GREATER
  | compare (fx, fx) = EQUAL
  | compare (fx, fy) = LESS
  | compare (fx, fh) = LESS
  | compare (fy, fz) = GREATER
  | compare (fy, fk) = GREATER
  | compare (fy, fx) = GREATER
  | compare (fy, fy) = EQUAL
  | compare (fy, fh) = LESS
  | compare (fh, fz) = GREATER
  | compare (fh, fk) = GREATER
  | compare (fh, fx) = GREATER
  | compare (fh, fy) = GREATER
  | compare (fh, fh) = LESS
		
end
end

structure V: VAR = struct
type var = variables
structure Ord: ORD_KEY = struct
type ord_key = var
fun compare (x1, x1) = EQUAL
  | compare (x1, x2) = LESS   
  | compare (x1, x3) = LESS
  | compare (x1, x4) = LESS
  | compare (x1, x5) = LESS
  | compare (x2, x1) = GREATER
  | compare (x2, x2) = EQUAL
  | compare (x2, x3) = LESS
  | compare (x2, x4) = LESS
  | compare (x2, x5) = LESS
  | compare (x3, x1) = GREATER
  | compare (x3, x2) = GREATER
  | compare (x3, x3) = EQUAL
  | compare (x3, x4) = LESS
  | compare (x3, x5) = LESS
  | compare (x4, x1) = GREATER
  | compare (x4, x2) = GREATER
  | compare (x4, x3) = GREATER
  | compare (x4, x4) = EQUAL
  | compare (x4, x5) = LESS
  | compare (x5, x1) = GREATER
  | compare (x5, x2) = GREATER
  | compare (x5, x3) = GREATER
  | compare (x5, x4) = GREATER
  | compare (x5, x5) = EQUAL
end
end

      
structure sigma = Term (structure S=S; structure V=V)
val map = sigma.empty

(* z ≡ z *)
val z = sigma.v x4

(*c is a constant*)
val c = sigma.validterm(fk,[])
		       
(* t1 = fx(x2,z)*)
val t1 = sigma.validterm(fx, [sigma.v x2, z])
			
(* t2 = fy(x3,c) where c is constant*)			
val t2 = sigma.validterm(fy, [sigma.v x3, c])
			
(* t3 = fh(x1) *)			
val t3 = sigma.validterm(fh, [sigma.v x1])
			
(* adding x1 ≡ t1 and x2 ≡ t2 *)
val map = valOf(sigma.addterm map (x1) t1)
val map = valOf(sigma.addterm map (x2) t2)

(* list of all variables direct and indirectly accessible with t3*)
val variablelist = sigma.varlist [] map (SOME t3)

(* val map = valOf(sigma.addterm map (x3) t3)) 

(* uncomment this to see that that it raises an error while trying to add x3 ≡ t3 to the map because
  x3 = fh(fx(fy(x3,c),z)) results in an infinite recursion
*)

*)
				 				  
	      

	       
