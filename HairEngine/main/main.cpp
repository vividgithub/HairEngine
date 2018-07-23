#include <iostream>

#include "../geo/hair.h"

int main() {
	using namespace HairEngine;
	using namespace std;

	Hair hair("C:\\Users\\VividWinPC1\\Developer\\Project\\HairEngine\\Houdini\\Resources\\Models\\Feamle 04 Retop\\Hair\\Straight-50000.hair");
	cout << hair;

	return 0;
}