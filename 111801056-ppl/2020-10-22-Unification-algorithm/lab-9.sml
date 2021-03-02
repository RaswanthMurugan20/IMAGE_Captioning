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
	if checker x (varlist [] tele (SOME(t))) = true then (SOME (Telescope.insert(tele,x,t)))
			     else NONE					      			
    end	  
	
end
   
functor Unify (structure S:SIG; structure V:VAR) = struct

structure tm = Term (structure S=S; structure V=V)
type term = tm.term
structure Telescope = tm.Telescope			

(*commbining 2 term lists into on [s1,s2...] [t1,t2,....] => [(s1,t1),(s2,t2),....] *)
fun comblist ((x::xs),(y::ys)) =  [(x,y)] @ (comblist (xs,ys))
  | comblist ([],[]) = []
  | comblist (xs,[]) = []
  | comblist ([],ys) = [] 

(*unify and unifylist functions*)
	
fun unify (tele: term Telescope.map) (s: term, t: term): term Telescope.map option =
    case (s,t) of
	(tm.v(x),tm.v(y)) => (case (Telescope.find(tele,x), Telescope.find(tele,y)) of
				(SOME a, SOME b) => unify tele (a,b)
			       |(NONE, NONE) => tm.addterm tele x (tm.v(y))
			       |(SOME b, NONE) => tm.addterm tele y b
			       |(NONE, SOME a) => tm.addterm tele x a)
      | (tm.ft(f,ts),tm.v(x)) => (case (Telescope.find(tele, x)) of
					  NONE => tm.addterm tele x (tm.ft(f,ts))
					| SOME a => unify tele (a,tm.ft(f,ts)) )
      | (tm.v(x),tm.ft(f,ts)) => unify tele (tm.ft(f,ts), tm.v(x)) 

			
      | (tm.ft(f,(xs)),tm.ft(g,(ys))) => if (S.Ord.compare(f,g) = EQUAL andalso (List.length (xs) = List.length (ys))) then unifylist tele (comblist (xs,ys)) else NONE 


and unifylist (tele: term Telescope.map) (ls: (term*term) list) =
    case (ls) of
	(x::xs) => let val temp = (unify tele x)
		   in case (temp) of
			  NONE => NONE
			| SOME a => unifylist a xs
		   end
      | [] => (SOME tele)


		    
end


				 
