//
// Created by vivi on 07/02/2018.
//

#pragma once

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <type_traits>

/*
 * Safe Delete
 */
#define HairEngine_SafeDelete(item_) if(item_ != nullptr) delete item_
#define HairEngine_SafeDeleteArray(item_) delete [] item_
#define HairEngine_AllocatorAllocate(item_, n_) item_ = std::allocator<std::remove_pointer<decltype(item_)>::type>().allocate(n_)
#define HairEngine_AllocatorDeallocate(item_, n_) if (item_) std::allocator<std::remove_pointer<decltype(item_)>::type>().deallocate(item_, n_)

/*
 * Some useful definitions for debugging so that we could access all the private data in the class
 * */
#if defined(NDEBUG) && !defined(HAIRENGINE_TEST)
	#define HairEngine_Public public
	#define HairEngine_Protected protected
	#define HairEngine_Private private
#else
	#define HairEngine_Public public
	#define HairEngine_Protected public
	#define HairEngine_Private public
#endif

/*
 * Debug
 */
#ifdef NDEBUG
#define HairEngine_DebugIf if(0)
#else
#define HAIRENGINE_DEBUG_ON
#define HairEngine_DebugIf if(1)
#endif

/*
 * VPbrt Enable
 */
#ifdef HAIRENGINE_ENABLE_VPBRT
#define HairEngine_VPbrtIf if(1)
#endif

/*
* Assertion
*/
#ifdef NDEBUG
#define HairEngine_DebugAssert(expr_) ((void)0)
#else
#define HairEngine_DebugAssert(expr_) assert(expr_)
#endif
