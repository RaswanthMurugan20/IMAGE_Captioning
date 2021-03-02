datatype lam = V of string
    | A of lam * lam
    | L of string * lam

datatype hlam = HV of string
    | HA of hlam * hlam
    | HL of hlam -> hlam

(* Q.1

Define the substitution function subst : string * hlam -> hlam -> hlam. The expression subst 
("x",u) e should replace every free occurance of the variable "x" in e with u. The tricky case is how 
does one handle the HL f case. HINT subst (x,u) (HL f) is HL fp where fp inp is essentially f inp but 
for substituting x by u. 

*)

fun subst (x, y) (HV e) =
    if e = x then y else HV e
    | subst (x, y) (HA (e1, e2)) = HA (subst (x, y) e1, subst (x, y) e2)
    | subst (x, y) (HL f) = 
                let
                  fun fp t = subst (x, y) (f t)
                in
                  HL fp
                end

(* Q.2 

Notice that the substitution did not have to worry about the variable bound by the lambda and is 
simpler than the one defined for standard lambda calculus representation. However building a lambda 
term is difficult. Define the function abstract : string -> hlam -> hlam that essentially builds the 
HOAS term for λ x . M, i.e. if mh : hlam is the HOAS representation of M, then abstract "x" mh should 
give the HOAS representation for λ x . M. HINT The HOAS for λ x . M is given HL f, where f is that 
function that takes a HOAS nh : hlam and gives the HOAS term obtained by substituting "x" with nh in 
mh. Use part 2 to complete the assignment.

*)

fun abstract q m = 
    let
      fun f n = subst (q, n) m
    in
      HL f
    end

(* Q.3 

Define a function freshen : hlam -> string which will generate a string that does not occur free in 
its input. For example, for something like HL (fn t => HA (t , HV "x")) it can generate any string 
but the string "x". If we have a way to compute the free variables of hlam one can use that. However, 
the freshen is easier. free : hlam -> list string. Write a helper function freeP : hlam -> list 
string such that for all t : hlam freeP t ∩ FV(t) = ∅. Idea is when one sees FV(HAb f) ⊂ f 
(HV "x").Define a function freshen : hlam -> string which will generate a string that does not occur 
free in its input. For example, for something like HL (fn t => HA (t , HV "x")) it can generate any 
string but the string "x". If we have a way to compute the free variables of hlam one can use that. 
However, the freshen is easier. free : hlam -> list string. Write a helper function 
freeP : hlam -> list string such that for all t : hlam freeP t ∩ FV(t) = ∅. Idea is when one sees 
FV(HAb f) ⊂ f (HV "x").

*)

 fun util [] i = ""
   | util (x::xs) i =
     let
         fun CharReturn x i =
             if (i >= 0) andalso (i < (String.size x))		     
             then
                 if String.sub(x, i) = #"p"		   
                 then "q"
                 else "p" 
             else "p"
     in
         (CharReturn x i) ^ (util xs (i+1))			
     end
	 

fun free [] = "p"
  | free ls = util ls 0
			 
fun freeP (HV e1) = [e1]
            | freeP (HA (e1, e2)) = (freeP e1) @ (freeP e2)
            | freeP (HL f) = freeP (f (HV "x"))

fun freshen e =  free (freeP e)

(* Q.4 

Give a conversion functions hoas : lam -> hlam syntax : hlam -> lam. In both cases handling the
constructor associated with with lambda abstraction is difficult.

*)

fun hoas (V e1) = HV e1
    | hoas (A (e1, e2)) = HA (hoas e1, hoas e2)
    | hoas (L (x, e1)) = abstract x (hoas e1)

fun syntax (HV e1) = V e1
    | syntax (HA (e1, e2)) = A (syntax e1, syntax e2)
    | syntax (HL f) = 
        let
          val makevar = freshen (HL f)
        in
          L (makevar, syntax (f (HV makevar)))
        end
