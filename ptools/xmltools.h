#ifndef XMLTOOLS_H_
#define XMLTOOLS_H_

// The tools here allow us to 
// (1) translate from XML to Val/Tar/Arr using xmlreader.h
// (2) translate from Val/Tab/Arr to XML using xmldumper.h
//
// The basic premise of these tools is that you are using
// XML as key-value pairs, and thus translating between
// dictionaries and XML is straight-forward.  In the
// following example, there is an obvious mapping between 
// dictionaries and XML:
//
//   <book attr1="1" attr2="2">
//      <chapter>text chap 1</chapter>
//      <chapter>text chap 2</chapter>
//   </book>
// 
// ----------------------------------
// 
//  { 'book' = {
//        '__attrs__' = { 'attr1':"1", 'attr2':"2" }
//        'chapter' = [ 'text chap1', 'text chap2']
//  }
//
// Adding attributes complicates the issues: many of the options
// below help control how the attributes in XML gets translated.
// The examples below showing UNFOLDING (or not) of attributes
// 
// <html>
//   <book attr1="1" attr2="2">
//     <chapter> chapter 1 </chapter>
//     <chapter> chapter 2 </chapter>
//   </book>
// </html>
// ----------------------------------------
// { 'html': {           
//      'book': {
//         '_attr1':"1",     <!--- Attributes UNFOLDED -->
//         '_attr2':"2",
//         'chapter': [ 'chapter1', 'chapter2' ]
//      }
// }
//  or
// { 'html' : {
//      'book': {
//         '__attrs__': { 'attr1'="1", 'attr2'="2" },  <!-- DEFAULT way -->
//         'chapter' : [ 'chapter1', 'chapter2' ]
//      }
//   }
// }


// ** Example where XML really is better:
// ** This is more of a "document", where HTML is better (text and
// key-values are interspersed)
// <html>
//   <book attr1="1" attr2="2">
//     This is the intro
//     <chapter> chapter 1 </chapter>
//     This is the extro
//   </book>
// </html>
// 
// {
//   'book': { 
//      'chapter': { ???
//      }
//   }
// }
// 
// ???? ['book'] -> "This is the intro" or "This is the outro?"
// NEITHER.  It gets dumped, as book is a dictionary.
// This is an example where converting from XML to Dictionaries
// may be a bad idea and just may not be a good correspondance.


// Options are formed by 'or'ing together. 

#include "xmldumper.h"
#include "xmlloader.h"


#endif // XMLTOOLS_H_
