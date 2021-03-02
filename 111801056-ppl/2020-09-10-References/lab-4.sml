(*Q1*)
signature COUNTER = sig
    val get: unit -> int
    val inc: unit -> unit
    val decrement: unit -> unit		 
end

structure Counter :COUNTER = struct
 val q = ref 0
 fun new_start x = q := x  (* starting counter from x *)	    
 fun get ()= !q
 fun inc ()= (q := !q + 1)
 fun decrement ()= (q := !q - 1)                   		      
end

(*Q2*)
functor MkCounter () :COUNTER = struct
 val q = ref 0   	    
 fun get ()= !q
 fun inc ()= (q := !q + 1)
 fun decrement ()= (q := !q - 1)
 fun new_start x = q := x (* starting counter from x *)
		       
end

				   
	       
		     
			
    
